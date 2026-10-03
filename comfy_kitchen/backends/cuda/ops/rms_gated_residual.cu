// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace comfy {
namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ __launch_bounds__(128) void rms_gated_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ activation,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    int width, float eps) {
    __shared__ float warp_sums[4];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * width;

    float sum_sq = 0.0f;
    // Match PyTorch CUDA RMSNorm's four values per thread and four-warps
    // reduction. HIP uses eight waves, so its merge order is not interchangeable.
    for (int col = tid * 4; col < width; col += 128 * 4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float value = __bfloat162float(activation[row_offset + col + i]);
            sum_sq = __fmaf_rn(value, value, sum_sq);
        }
    }
    sum_sq = warp_sum(sum_sq);
    #pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
        if (lane == 0 && warp >= offset && warp < 2 * offset) {
            warp_sums[warp - offset] = sum_sq;
        }
        __syncthreads();
        if (lane == 0 && warp < offset) sum_sq += warp_sums[warp];
        __syncthreads();
    }
    if (tid == 0) warp_sums[0] = rsqrtf(__fdiv_rn(sum_sq, float(width)) + eps);
    __syncthreads();
    const float rstd = warp_sums[0];

    for (int col = tid; col < width; col += blockDim.x) {
        // Match the separate eager kernels' visible BF16 rounding points.
        float normalized = __fmul_rn(rstd, __bfloat162float(activation[row_offset + col]));
        normalized = round_bf16(__fmul_rn(normalized, __bfloat162float(norm_weight[col])));
        const float product = round_bf16(normalized * __bfloat162float(gate[col]));
        output[row_offset + col] = __float2bfloat16_rn(
            __bfloat162float(residual[row_offset + col]) + product);
    }
}

} // namespace
} // namespace comfy

extern "C" void launch_rms_gated_residual_bf16_kernel(
    const void* activation, const void* norm_weight, const void* residual,
    const void* gate, void* output, int rows, int width, float eps,
    cudaStream_t stream) {
    if (rows < 0 || width <= 0 || width % 4 != 0) {
        throw std::runtime_error(
            "rms_gated_residual_bf16 requires non-negative rows and positive width divisible by 4");
    }
    if (rows == 0) return;
    comfy::rms_gated_residual_bf16_kernel<<<rows, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation),
        static_cast<const __nv_bfloat16*>(norm_weight),
        static_cast<const __nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<__nv_bfloat16*>(output), width, eps);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA RMS gated residual failed: ") +
                                 cudaGetErrorString(error));
    }
}
