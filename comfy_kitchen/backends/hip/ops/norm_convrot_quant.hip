// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// BF16 normalization/modulation -> the original scalar ConvRot-256 producer ->
// rowwise INT8 quantization.  Keep this separate from the packed wide-row
// producer: its different butterfly schedule can change INT8 rounding.
#include <stdexcept>

#include "../hadamard.h"

namespace comfy::hip_backend {

__global__ __launch_bounds__(256) void norm_convrot_quant_kernel(
    const __bf16* __restrict__ x, const float* __restrict__ weight,
    const void* __restrict__ scale, const void* __restrict__ shift,
    int scale_code, int shift_code, int8_t* __restrict__ qout,
    float* __restrict__ scaleout, int K, float eps, int norm_kind,
    bool tiled_output) {

    constexpr int threads = 256;
    __shared__ float red[threads];
    __shared__ float transform[4 * 2 * 256];
    extern __shared__ __bf16 normalized[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * K;
    const float inv_k = 1.0f / static_cast<float>(K);

    float mean = 0.0f;
    if (norm_kind == 1) {  // AdaLN: exactly the scalar adaln.hip reduction.
        float sum = 0.0f;
        for (int i = t; i < K; i += threads) sum += static_cast<float>(x[base + i]);
        red[t] = sum;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (t < s) red[t] += red[t + s];
            __syncthreads();
        }
        mean = red[0] * inv_k;
    }

    float sq = 0.0f;
    float rstd;
    if (norm_kind == 0) {
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(on)
        #pragma clang fp reciprocal(off)
        // PyTorch's BF16 RMS implementation: four adjacent values per thread,
        // wave32 reduction, then the same 8-wave 4/2/1 inter-wave tree.
        for (int v = t; v < K / 4; v += threads) {
            const convrot_bf16x4 values =
                reinterpret_cast<const convrot_bf16x4*>(x + base)[v];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float value = static_cast<float>(values[e]);
                sq += value * value;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sq += __shfl_down(sq, offset, 32);
        if ((t & 31) == 0) red[t >> 5] = sq;
        __syncthreads();
        if (t < 4) red[t] += red[t + 4];
        __syncthreads();
        if (t < 2) red[t] += red[t + 2];
        __syncthreads();
        if (t == 0) red[0] += red[1];
        __syncthreads();
        rstd = __ocml_rsqrt_f32(ieee_div_f32(red[0], static_cast<float>(K)) + eps);
    } else {
        for (int i = t; i < K; i += threads) {
            const float d = static_cast<float>(x[base + i]) - mean;
            sq += d * d;
        }
        red[t] = sq;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (t < s) red[t] += red[t + s];
            __syncthreads();
        }
        rstd = rsqrtf(red[0] * inv_k + eps);
    }

    auto normalize = [&](int i) {
        const float value = static_cast<float>(x[base + i]);
        if (norm_kind == 0) {
            float out = value * rstd;
            asm volatile("" : "+v"(out));
            out *= weight[i];
            if (scale) out *= 1.0f + load_in(scale, i, scale_code);
            if (shift) out += load_in(shift, i, shift_code);
            return out;
        } else {
            // Retain the separate AdaLN operation's BF16 output contract.
            const float s = scale ? load_in(scale, i, scale_code) : 0.0f;
            const float h = shift ? load_in(shift, i, shift_code) : 0.0f;
            return round_bf16(((value - mean) * rstd) * (1.0f + s) + h);
        }
    };

    float lmax = 0.0f;
    const int groups = K / 256;
    const int sub = t / 64;
    const int lane = t % 64;
    float* buf0 = transform + sub * 512;
    float* buf1 = buf0 + 256;
    for (int group_base = 0; group_base < groups; group_base += 4) {
        const int group = group_base + sub;
        const bool active = group < groups;
        const int column = lane * 4;
        float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
        if (active) {
            const int i = group * 256 + column;
            v0 = normalize(i);
            v1 = normalize(i + 1);
            v2 = normalize(i + 2);
            v3 = normalize(i + 3);
        }
        buf1[column] = 0.5f * (v0 + v1 + v2 - v3);
        buf1[column + 1] = 0.5f * (v0 + v1 - v2 + v3);
        buf1[column + 2] = 0.5f * (v0 - v1 + v2 + v3);
        buf1[column + 3] = 0.5f * (-v0 + v1 + v2 + v3);
        __syncthreads();
        convrot_fht_stage64<4>(buf1, buf0, lane);
        __syncthreads();
        convrot_fht_stage64<16>(buf0, buf1, lane);
        __syncthreads();
        if (active) {
            lmax = fmaxf(lmax, convrot_fht_stage64_store_absmax_typed<64, __bf16>(
                buf1, normalized + group * 256, lane));
        }
        __syncthreads();
    }
    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float rowmax = fmaxf(finite_absmax_for_quant<__bf16>(red[0]), 1e-10f);
    const float qscale = rowmax / 127.0f;
    if (t == 0) scaleout[row] = qscale;
    float inv = 127.0f / rowmax;
    asm volatile("" : "+v"(inv));
    for (int i = t; i < K; i += threads) {
        int q = static_cast<int>(rintf(static_cast<float>(normalized[i]) * inv));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        const int64_t qi = tiled_output
            ? static_cast<int64_t>(row >> 7) * K * 128 +
              static_cast<int64_t>(i >> 7) * 128 * 128 + (row & 127) * 128 + (i & 127)
            : base + i;
        qout[qi] = static_cast<int8_t>(q);
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_norm_convrot_quant_kernel(
    const void* x, const void* weight, const void* scale, int scale_code,
    const void* shift, int shift_code, void* q, void* scales, int M, int K,
    float eps, int norm_kind, bool tiled_output, hipStream_t stream) {
    if (M == 0) return;
    if (K <= 0 || K % 256 != 0 || (norm_kind != 0 && norm_kind != 1))
        throw std::runtime_error("norm_convrot_quant: invalid shape or norm kind");
    comfy::hip_backend::norm_convrot_quant_kernel<<<
        M, 256, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
        static_cast<const __bf16*>(x), static_cast<const float*>(weight), scale, shift,
        scale_code, shift_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
        K, eps, norm_kind, tiled_output);
}
