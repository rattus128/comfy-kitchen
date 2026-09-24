// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// W4A8 weight dequant: grouped int4 -> int8 for the tuned int8-GEMM path.
//
// AsymW4A8Int8Layout dequantizes int4 weights to "grouped int8" (per-group scale
// folded in, per-channel scale left for the int8 GEMM epilogue), then runs comfy's
// tuned int8 CUTLASS GEMM. So this file is just the memory-bound int4->int8 dequant
// kernel (fp32/fp8-e4m3 group scales, optional codebook); the matmul is cutlass_gemm_int8.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"
#include "../prefetch_ring.h"

// Grouped int4 -> int8 dequant for the int8-GEMM W4A8 path: out[n,k] =
// round((q_u[n,k]-8) * s_rel[n, k/G]), q_u packed uint4 (even col=low nibble).
// s_rel = per-group scale / per-channel scale (so the int8 range is used). The
// per-channel scale is applied later in the int8 GEMM epilogue. Memory-bound.
namespace {
__device__ PrefetchRingState* g_w4a8_prefetch_ring = nullptr;

// Per-group scale is fp32 or fp8 (e4m3). fp8 halves the scale metadata at a tiny
// quality cost. uint8_t storage == e4m3 raw bits.
template <typename ScaleT> __device__ __forceinline__ float load_scale(ScaleT v);
template <> __device__ __forceinline__ float load_scale<float>(float v) { return v; }
template <> __device__ __forceinline__ float load_scale<uint8_t>(uint8_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

template <typename T> __device__ __forceinline__ T store_output(float value);
template <> __device__ __forceinline__ float store_output<float>(float value) { return value; }
template <> __device__ __forceinline__ __half store_output<__half>(float value) { return __float2half(value); }
template <> __device__ __forceinline__ __nv_bfloat16 store_output<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

// Decode one uint2 (8 packed bytes = 16 int4 codes, low nibble = even col) to
// 16 int8 on the __float2int_rn(level * scale) grid shared by every W4A8 path.
// cb is the 16-entry level table or nullptr for the uniform (q-8) levels.
// sc0..sc3 are the up-to-4 distinct group scales the 16 cols can span; all
// four are equal when G >= 16.
__device__ __forceinline__ void dequant16_int4_to_int8(
    uint2 pk, const float* __restrict__ cb,
    float sc0, float sc1, float sc2, float sc3, int G, char4 out4[4])
{
    const unsigned words[2] = {pk.x, pk.y};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        #pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
            const int oo = w * 4 + bi;             // 0..7 -> cols oo*2, oo*2+1
            const int lg = (G >= 16) ? 0 : ((oo * 2) / G);  // local group in the vec
            const float s = (lg == 0) ? sc0 : (lg == 1 ? sc1 : (lg == 2 ? sc2 : sc3));
            const unsigned byte = (words[w] >> (bi * 8)) & 0xFF;
            const unsigned c0 = byte & 0xF, c1 = (byte >> 4) & 0xF;
            const float v0 = cb ? cb[c0] : (static_cast<float>(c0) - 8.0f);
            const float v1 = cb ? cb[c1] : (static_cast<float>(c1) - 8.0f);
            reinterpret_cast<int8_t*>(&out4[oo / 2])[(oo % 2) * 2]     =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(v0 * s))));
            reinterpret_cast<int8_t*>(&out4[oo / 2])[(oo % 2) * 2 + 1] =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(v1 * s))));
        }
    }
}

// Each thread: 8 packed bytes (uint2) -> 16 int8 (uint4 store). The 16 output
// cols may span multiple groups when G<16 (finer groups = better int4 quality),
// so the scale is (re)loaded per output pair from its own group. Only 4 group
// scales (sc0..sc3) are loaded, so a 16-col vec may span at most 4 groups: G must
// be in {4, 8, 16} or a multiple of 16 (G<4 would span >4 groups and mis-scale).
// If codebook != nullptr, the 4-bit code indexes a shared 16-entry non-uniform
// codebook (Lloyd-Max on the rotated-Gaussian weight) instead of the uniform
// level (q-8); same storage/speed, ~14% lower weight error at coarse groups.
template <typename ScaleT>
__global__ void dequant_int4_grouped_to_int8_kernel(
    const int8_t* __restrict__ qw,   // (N, K/2) packed uint4
    const ScaleT* __restrict__ s_rel,// (N, K/G) fp32 or e4m3 raw
    const float*  __restrict__ codebook, // 16 floats or nullptr
    int8_t*       __restrict__ out,  // (N, K)
    long n_vec, int Khalf, int K, int G)
{
    __shared__ float cb[16];
    if (codebook && threadIdx.x < 16) cb[threadIdx.x] = codebook[threadIdx.x];
    if (codebook) __syncthreads();
    long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vec) return;                       // n_vec = N*Khalf/8
    const int vec_per_row = Khalf / 8;
    const int n = v / vec_per_row;
    const int hv = v % vec_per_row;               // which uint2 in the row
    const int kh = hv * 8;                        // packed byte offset
    const int k0 = kh * 2;                         // output col base (16 wide)
    const int nG = K / G;
    const long srow = (long)n * nG;
    const uint2 pk = *reinterpret_cast<const uint2*>(&qw[(long)n * Khalf + kh]);
    // The 16-col vec spans 1 group (G>=16, the common case), 2 (G=8), or 4 (G=4).
    // Load+decode each distinct group scale ONCE instead of per output pair.
    const int base_g = k0 / G;
    float sc0 = load_scale<ScaleT>(s_rel[srow + base_g]);
    float sc1 = sc0, sc2 = sc0, sc3 = sc0;
    if (G < 16) {
        sc1 = load_scale<ScaleT>(s_rel[srow + base_g + 1]);
        if (G < 8) {  // G == 4
            sc2 = load_scale<ScaleT>(s_rel[srow + base_g + 2]);
            sc3 = load_scale<ScaleT>(s_rel[srow + base_g + 3]);
        }
    }
    char4 o4[4];
    dequant16_int4_to_int8(pk, codebook ? cb : nullptr, sc0, sc1, sc2, sc3, G, o4);
    *reinterpret_cast<uint4*>(&out[(long)n * K + k0]) = *reinterpret_cast<uint4*>(o4);
}

__device__ __forceinline__ unsigned decode_w4a8_lut4(
    unsigned codes, unsigned scale_bits, const int8_t* __restrict__ decode_lut)
{
    const int8_t* __restrict__ table = decode_lut + scale_bits * 16;
    unsigned decoded = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned code = (codes >> (i * 4)) & 0xfu;
        decoded |= static_cast<unsigned>(static_cast<uint8_t>(table[code])) << (i * 8);
    }
    return decoded;
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int (&acc)[4], const unsigned (&a)[4], const unsigned (&b)[2])
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
}

// Interpret the usual row-major [tokens, K] activation as column-major [K, tokens]
// and compute weight[N, K] @ activation[K, tokens]. One warp produces a 16-output
// by 8-token MMA tile. Split-K restores enough blocks to fill large GPUs; each
// split atomically contributes exact INT32 partials to the small output workspace.
// Packed W4 is decoded directly into tensor-core registers, never a weight workspace.
template <int WarpsPerBlock>
__global__ void w4a8_codebook_mma_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const int8_t* __restrict__ decode_lut,
    int* __restrict__ workspace,
    int M, int N, int K, int G, int split_k)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int n0 = (static_cast<int>(blockIdx.x) * WarpsPerBlock + warp) * 16;
    const int k_per_split = K / split_k;
    const int split = static_cast<int>(blockIdx.y);
    const int k_begin = split * k_per_split;
    const int k_end = k_begin + k_per_split;
    const int tiles = (N + 15) / 16;
    const int64_t first_record =
        (static_cast<int64_t>(split) * tiles + n0 / 16) * (k_per_split / 32);

    int acc[4] = {};
    for (int k0 = k_begin; k0 < k_end; k0 += 32) {
        const int k_lane = thread_in_group * 4;
        const int64_t tile = first_record + (k0 - k_begin) / 32;
        const int8_t* __restrict__ tile_data = weight + tile * 288;
        const uint2 packed = reinterpret_cast<const uint2*>(tile_data)[lane];
        unsigned scales = 0;
        if (thread_in_group == 0) {
            scales = reinterpret_cast<const unsigned*>(tile_data + 256)[group];
        }
        scales = __shfl_sync(0xffffffffu, scales, group * 4);
        const unsigned a[4] = {
            decode_w4a8_lut4(packed.x & 0xffffu, scales & 0xffu, decode_lut),
            decode_w4a8_lut4(packed.x >> 16, (scales >> 8) & 0xffu, decode_lut),
            decode_w4a8_lut4(packed.y & 0xffffu, (scales >> 16) & 0xffu, decode_lut),
            decode_w4a8_lut4(packed.y >> 16, scales >> 24, decode_lut),
        };
        const int token = group;
        const int* __restrict__ x4 = token < M
            ? reinterpret_cast<const int*>(x + static_cast<int64_t>(token) * K)
            : nullptr;
        const unsigned b[2] = {
            x4 ? static_cast<unsigned>(x4[(k0 + k_lane) / 4]) : 0u,
            x4 ? static_cast<unsigned>(x4[(k0 + k_lane + 16) / 4]) : 0u,
        };
        mma_m16n8k32_s8(acc, a, b);
    }
    // Ring consumption: one add per block for the records its warps streamed.
    if (threadIdx.x == 0) {
        const int block_tiles = min(WarpsPerBlock, tiles - static_cast<int>(blockIdx.x) * WarpsPerBlock);
        prefetch_ring_consume_device(
            g_w4a8_prefetch_ring,
            static_cast<uint64_t>(block_tiles) * (k_per_split / 32) * 288);
    }

    const int token0 = thread_in_group * 2;
    const int token1 = token0 + 1;
    const int n_top = n0 + group;
    const int n_bottom = n_top + 8;
    if (n_top < N) {
        if (token0 < M) {
            atomicAdd(&workspace[static_cast<int64_t>(token0) * N + n_top], acc[0]);
        }
        if (token1 < M) {
            atomicAdd(&workspace[static_cast<int64_t>(token1) * N + n_top], acc[1]);
        }
    }
    if (n_bottom < N) {
        if (token0 < M) {
            atomicAdd(&workspace[static_cast<int64_t>(token0) * N + n_bottom], acc[2]);
        }
        if (token1 < M) {
            atomicAdd(&workspace[static_cast<int64_t>(token1) * N + n_bottom], acc[3]);
        }
    }
}

// Decode one 16-code chunk (one scale byte) with the LUT row for that scale held in
// shared memory: two byte permutes pick codes 0-7 / 8-15, code bit 3 selects between
// them. Same LUT bytes as decode_w4a8_lut4, so the result is bit-identical.
__device__ __forceinline__ unsigned decode_w4a8_chunk(
    unsigned codes, unsigned scale_bits, const uint4* __restrict__ lut)
{
    const uint4 table = lut[scale_bits];
    // selector nibbles use bits 0-2 only; bit 3 of each code then picks lo/hi per byte
    const unsigned lo = __byte_perm(table.x, table.y, codes);
    const unsigned hi = __byte_perm(table.z, table.w, codes);
    return __byte_perm(lo, hi, ((codes >> 1) & 0x4444u) | 0x3210u);
}

__device__ __forceinline__ void decode_w4a8_record_smem(
    const uint2& packed, unsigned scales,
    const uint4* __restrict__ lut, unsigned (&decoded)[4])
{
    decoded[0] = decode_w4a8_chunk(packed.x, scales & 0xffu, lut);
    decoded[1] = decode_w4a8_chunk(packed.x >> 16, (scales >> 8) & 0xffu, lut);
    decoded[2] = decode_w4a8_chunk(packed.y, (scales >> 16) & 0xffu, lut);
    decoded[3] = decode_w4a8_chunk(packed.y >> 16, scales >> 24, lut);
}

__device__ __forceinline__ void cp_async16_evict_first(uint32_t smem, const void* gptr) {
#ifdef W4A8_NO_EVICT_FIRST
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gptr));
#else
    // Weights are read once per step: evict_first keeps the demand stream from
    // displacing lines the prefetch ring already landed; ring-prefetched lines
    // (inserted with the default priority) still hit.
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(smem), "l"(gptr), "l"(policy));
#endif
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Streaming decode GEMM for the Qwen schedule. A warp owns one 16-output tile over
// `rows` consecutive K records and streams them through a Stages-deep per-warp
// cp.async ring in shared memory: one record is decoded + MMA'd while Stages-1 are
// in flight, for the warp's whole life. Register rings can't do this: ptxas gives
// every load in the loop one scoreboard, so waiting on one record waits on all.
//
// Weight layout is the checkpoint's split-major packing with PackRows records per
// (split, tile): record for K-row `krow` of `tile` lives at
// ((krow / PackRows) * tiles + tile) * PackRows + krow % PackRows. `rows` is
// decoupled from PackRows (chosen by the launcher for occupancy), so a warp's
// records are rows/PackRows runs of PackRows*288 contiguous bytes.
//
// LUT and the block's x slice are staged in shared memory after the first loads are
// issued. The split-K reduction finishes in-kernel: each warp adds its tile into the
// int32 workspace, bumps the tile's counter, and the warp that arrives last reads
// the tile back from L2, applies the scales/bias, writes the output, and returns
// the workspace tile and counter to zero. So the workspace and counters are zero
// on entry and on exit, and the caller keeps them across launches.
#ifndef W4A8_STREAM_STAGES
#define W4A8_STREAM_STAGES 4
#endif
constexpr int kStreamStages = W4A8_STREAM_STAGES;
constexpr int kStreamMaxRows = 32;

__host__ __device__ constexpr int w4a8_stream_smem_bytes(int warps_per_block, int rows) {
    return 4096 + 8 * (rows * 32 + 16) + warps_per_block * kStreamStages * 288;
}

#ifdef W4A8_BLOCK_TRACE
// Diagnostic only (harness builds): per-block smid and globaltimer start/end of the stream kernel.
struct W4A8BlockTrace { unsigned sm, pad; unsigned long long t0, t1; };
__device__ W4A8BlockTrace* g_w4a8_block_trace = nullptr;
#define W4A8_TRACE_BEGIN() unsigned long long _tr_t0 = 0; unsigned _tr_sm = 0; \
    if (threadIdx.x == 0 && g_w4a8_block_trace) { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_tr_t0)); asm("mov.u32 %0, %%smid;" : "=r"(_tr_sm)); }
#define W4A8_TRACE_END() if (threadIdx.x == 0 && g_w4a8_block_trace) { unsigned long long _t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t1)); \
    W4A8BlockTrace& r = g_w4a8_block_trace[blockIdx.y * gridDim.x + blockIdx.x]; r.sm = _tr_sm; r.t0 = _tr_t0; r.t1 = _t1; }
#else
#define W4A8_TRACE_BEGIN()
#define W4A8_TRACE_END()
#endif

template <int WarpsPerBlock, int PackRows, typename OutputT>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void w4a8_codebook_mma_stream_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const int8_t* __restrict__ decode_lut,
    const float* __restrict__ s_channel,
    const float* __restrict__ x_scales,
    const float* __restrict__ bias,
    int* __restrict__ workspace,
    int* __restrict__ counters,
    OutputT* __restrict__ output,
    int M, int N, int K, int rows)
{
    constexpr int S = kStreamStages;
    W4A8_TRACE_BEGIN();
    extern __shared__ __align__(16) uint8_t smem[];
    uint4* lut = reinterpret_cast<uint4*>(smem);
    uint8_t* xs_s = smem + 4096;
    const int x_stride = rows * 32 + 16;   // +16 keeps the per-token rows off one bank pattern
    uint8_t* stages = xs_s + 8 * x_stride;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tiles = N / 16;
    const int output_tile = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
    const bool active = output_tile < tiles;
    const int split = static_cast<int>(blockIdx.y);
    const int k_begin = split * rows * 32;

    // lanes 0..17 each move 16 B of the 288 B record
    const int8_t* tile_base = weight
        + static_cast<int64_t>(output_tile) * PackRows * 288 + lane * 16;
    const int64_t chunk_stride = static_cast<int64_t>(tiles) * PackRows * 288;
    auto record = [&](int i) -> const int8_t* {
        const int krow = split * rows + i;
        return tile_base + (krow / PackRows) * chunk_stride + (krow % PackRows) * 288;
    };
    const bool issuer = active && lane < 18;
    const uint8_t* my_stages = stages + warp * S * 288;
    const uint32_t st_s = static_cast<uint32_t>(__cvta_generic_to_shared(my_stages)) + lane * 16;

    // Ring consumption: thread 0 credits the block's PackRows*288-byte runs as
    // its own loads of them complete (sibling warps stream in lockstep, so the
    // error is < 1 block). The ring's byte stream is the host-recorded chunk
    // order, k outer and split inner (w4a8_stream_chunks), and every block of
    // a wave passes each run boundary at about the same time, so the credited
    // total tracks the stream position.
    const uint64_t run_credit = static_cast<uint64_t>(
        min(WarpsPerBlock, tiles - static_cast<int>(blockIdx.x) * WarpsPerBlock)) * PackRows * 288;
    auto credit = [&](int i) {
        if (threadIdx.x == 0 && ((i + 1) & (PackRows - 1)) == 0)
            prefetch_ring_consume_device(g_w4a8_prefetch_ring, run_credit);
    };

    #pragma unroll
    for (int s = 0; s < S; ++s) {
        if (issuer) cp_async16_evict_first(st_s + s * 288, record(s));
        cp_async_commit();
    }
#ifdef W4A8_SELF_PREFETCH
    // Pull the warp's later runs into L2 now so the demand stream above hits.
    // Each run is PackRows contiguous records; run 0 is already in flight.
    if (active && lane == 0) {
        for (int r = PackRows; r < rows; r += PackRows) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                         :: "l"(record(r)), "r"(static_cast<unsigned>(PackRows * 288)) : "memory");
        }
    }
#endif
    for (int i = threadIdx.x; i < 256; i += WarpsPerBlock * 32) {
        lut[i] = reinterpret_cast<const uint4*>(decode_lut)[i];
    }
    for (int i = threadIdx.x; i < 8 * rows * 2; i += WarpsPerBlock * 32) {
        const int token = i / (rows * 2), v = i % (rows * 2);
        uint4 value = make_uint4(0u, 0u, 0u, 0u);
        if (token < M) {
            value = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(token) * K + k_begin)[v];
        }
        *reinterpret_cast<uint4*>(xs_s + token * x_stride + v * 16) = value;
    }
    __syncthreads();
    if (!active) return;

    const int token = lane >> 2;
    const int thread_in_group = lane & 3;
    const uint8_t* xrow = xs_s + token * x_stride + thread_in_group * 4;
    const uint8_t* rec_codes = my_stages + lane * 8;
    const uint8_t* rec_scales = my_stages + 256 + token * 4;
    int acc[4] = {};

    auto consume = [&](int stage, int i) {
        const uint2 packed = *reinterpret_cast<const uint2*>(rec_codes + stage * 288);
        const unsigned scales = *reinterpret_cast<const unsigned*>(rec_scales + stage * 288);
        unsigned decoded[4];
        decode_w4a8_record_smem(packed, scales, lut, decoded);
        const unsigned input[2] = {
            *reinterpret_cast<const unsigned*>(xrow + i * 32),
            *reinterpret_cast<const unsigned*>(xrow + i * 32 + 16),
        };
        mma_m16n8k32_s8(acc, decoded, input);
    };

    // Record i is consumed once at most S-1 newer groups are pending, then its stage
    // is refilled with record i+S. Unrolled by S so stage indices are compile-time.
    int i = 0;
    for (; i + S <= rows - S; i += S) {
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            cp_async_wait<S - 1>();
            __syncwarp();
            consume(j, i + j);
            credit(i + j);
            __syncwarp();
            if (issuer) cp_async16_evict_first(st_s + j * 288, record(i + S + j));
            cp_async_commit();
        }
    }
    for (; i < rows; ++i) {
        cp_async_wait<S - 1>();
        __syncwarp();
        consume(i % S, i);
        credit(i);
        __syncwarp();
        if (issuer && i + S < rows) cp_async16_evict_first(st_s + (i % S) * 288, record(i + S));
        cp_async_commit();
    }

    const int token0 = thread_in_group * 2;
    const int token1 = token0 + 1;
    const int n_top = output_tile * 16 + token;
    const int n_bottom = n_top + 8;
    int* ws00 = &workspace[static_cast<int64_t>(token0) * N + n_top];
    int* ws01 = &workspace[static_cast<int64_t>(token0) * N + n_bottom];
    int* ws10 = &workspace[static_cast<int64_t>(token1) * N + n_top];
    int* ws11 = &workspace[static_cast<int64_t>(token1) * N + n_bottom];
    if (token0 < M) {
        atomicAdd(ws00, acc[0]);
        atomicAdd(ws01, acc[2]);
    }
    if (token1 < M) {
        atomicAdd(ws10, acc[1]);
        atomicAdd(ws11, acc[3]);
    }

    // Publish this warp's adds, then arrive on the tile counter. The lane that sees
    // split_k-1 prior arrivals knows every split's adds are visible at L2.
    __threadfence();
    W4A8_TRACE_END();
    int prior = 0;
    if (lane == 0) prior = atomicAdd(&counters[output_tile], 1);
    prior = __shfl_sync(0xffffffffu, prior, 0);
    if (prior != static_cast<int>(gridDim.y) - 1) return;
    __threadfence();

    // Same lane->element mapping as the adds above, so the warp covers the whole
    // [M, 16] tile exactly once: read (L2), scale, store, and re-zero.
    auto finish = [&](int* ws, int tok, int n) {
        float value = static_cast<float>(__ldcg(ws)) * x_scales[tok] * s_channel[n];
        if (bias) value += bias[n];
        output[static_cast<int64_t>(tok) * N + n] = store_output<OutputT>(value);
        *ws = 0;
    };
    if (token0 < M) {
        finish(ws00, token0, n_top);
        finish(ws01, token0, n_bottom);
    }
    if (token1 < M) {
        finish(ws10, token1, n_top);
        finish(ws11, token1, n_bottom);
    }
    if (lane == 0) counters[output_tile] = 0;
}

template <typename OutputT>
__global__ void w4a8_codebook_mma_epilogue(
    const int* __restrict__ workspace,
    const float* __restrict__ s_channel,
    const float* __restrict__ x_scales,
    const float* __restrict__ bias,
    OutputT* __restrict__ output,
    int M, int N)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(M) * N) return;
    const int token = index / N;
    const int n = index - static_cast<int64_t>(token) * N;
    float value = static_cast<float>(workspace[index]) * x_scales[token] * s_channel[n];
    if (bias) value += bias[n];
    output[index] = store_output<OutputT>(value);
}
}  // namespace

extern "C" void set_w4a8_prefetch_ring_state(PrefetchRingState* state) {
    cudaMemcpyToSymbol(g_w4a8_prefetch_ring, &state, sizeof(state));
}

// codebook: 16 floats (non-uniform levels) or nullptr for uniform (q-8).
extern "C" void launch_dequant_int4_grouped_to_int8(
    const void* qw, const void* s_rel, const void* codebook, void* out,
    int64_t N, int64_t K, int64_t G, cudaStream_t stream)
{
    const int Khalf = K / 2;
    const long n_vec = (long)N * Khalf / 8;
    const int block = 256;
    const long grid = (n_vec + block - 1) / block;
    dequant_int4_grouped_to_int8_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const int8_t*>(qw), static_cast<const float*>(s_rel),
        static_cast<const float*>(codebook),
        static_cast<int8_t*>(out), n_vec, Khalf, static_cast<int>(K), static_cast<int>(G));
}

// fp8 (e4m3) per-group scale variant; s_rel passed as raw uint8 bits.
extern "C" void launch_dequant_int4_grouped_to_int8_e4m3(
    const void* qw, const void* s_rel, const void* codebook, void* out,
    int64_t N, int64_t K, int64_t G, cudaStream_t stream)
{
    const int Khalf = K / 2;
    const long n_vec = (long)N * Khalf / 8;
    const int block = 256;
    const long grid = (n_vec + block - 1) / block;
    dequant_int4_grouped_to_int8_kernel<uint8_t><<<grid, block, 0, stream>>>(
        static_cast<const int8_t*>(qw), static_cast<const uint8_t*>(s_rel),
        static_cast<const float*>(codebook),
        static_cast<int8_t*>(out), n_vec, Khalf, static_cast<int>(K), static_cast<int>(G));
}

// Fused-quality W4A8: dequant int4 -> int8 in column chunks (codebook + per-group
// s_rel) feeding the tuned STRIDED int8 GEMM, so each int8 weight chunk stays
// L2-resident instead of the full [N,K] round-tripping global (the convrot_w4a4
// chunking trick, run at our group-16 codebook quality). Returns false if the
// strided GEMM rejects a chunk config -> caller falls back to the 2-pass path.
// bias is read in the OUTPUT dtype (cutlass_gemm_int8.cu); never pass fp32 here.
extern "C" bool launch_cutlass_int8_dequant_strided(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int64_t output_stride, int out_dtype_code,
    cudaStream_t stream);

extern "C" bool launch_w4a8_codebook_gemm_chunked(
    const void* xq,        // [M, K] int8 activation
    const void* weight,    // [N, K/2] packed uint4
    const void* s_rel,     // [N, K/G] fp8 (e4m3) per-group scale
    const void* codebook,  // [16] fp32 or nullptr
    const void* s_channel, // [N] fp32 per-channel scale
    const void* xs,        // [M] fp32 per-row activation scale
    const void* bias,      // [N] in out_dtype, or nullptr
    void* workspace,       // [chunk_cols, K] int8 scratch (preallocated, reused)
    void* out,             // [M, N] output (out_dtype)
    int64_t M, int64_t N, int64_t K, int64_t G, int64_t chunk_cols,
    int out_dtype_code, cudaStream_t stream)
{
    // A non-positive chunk stride never advances n0 -> would loop forever; a non-positive
    // K/G would divide by zero below. Bail so the caller uses the 2-pass path.
    if (chunk_cols <= 0 || K <= 0 || G <= 0) return false;
    const int64_t Khalf = K / 2, KG = K / G, osz = (out_dtype_code == 0) ? 4 : 2;
    for (int64_t n0 = 0; n0 < N; n0 += chunk_cols) {
        const int64_t cols = (chunk_cols < N - n0) ? chunk_cols : (N - n0);
        launch_dequant_int4_grouped_to_int8_e4m3(
            static_cast<const int8_t*>(weight) + n0 * Khalf,
            static_cast<const uint8_t*>(s_rel) + n0 * KG,
            codebook, workspace, cols, K, G, stream);
        // bias is in the output dtype (the strided GEMM's contract), so it
        // advances by the same element size as the output.
        const void* bias_chunk = bias ? static_cast<const char*>(bias) + n0 * osz : nullptr;
        void* out_chunk = static_cast<char*>(out) + n0 * osz;
        if (!launch_cutlass_int8_dequant_strided(
                xq, workspace, xs, static_cast<const float*>(s_channel) + n0, bias_chunk,
                out_chunk, M, cols, K, N /*output_stride*/, out_dtype_code, stream))
            return false;
    }
    return true;
}

// Rows per warp the streaming kernel uses for an [N, K] weight packed with
// pack_rows records per split, or 0 when the generic kernel runs instead: the
// largest rows dividing K/32 that still fills about one wave of warps (halving
// otherwise), so small matrices (o_proj) do not run under-occupied. Exported so
// the host can record the kernel's read order for the prefetch ring.
extern "C" int w4a8_stream_rows(int64_t N, int64_t K, int64_t pack_rows) {
    if (N % 16 != 0 || (pack_rows != 8 && pack_rows != 16) || K % (pack_rows * 32) != 0) return 0;
    int device = 0, sm_count = 0;
    if (cudaGetDevice(&device) != cudaSuccess
            || cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return 0;
    const int64_t k_rows = K / 32;
    const int64_t tiles = N / 16;
    const int64_t wave = static_cast<int64_t>(sm_count) * 20;
    int rows = kStreamMaxRows;
    while (rows > pack_rows && (k_rows % rows != 0 || tiles * (k_rows / rows) < wave)) rows /= 2;
    return rows;
}

// workspace [M, N] int32 and counters [N/16] int32 must be zero on entry; the streaming
// kernel leaves them zero, and the generic kernel re-zeroes the workspace after its
// epilogue, so the caller can keep both across launches.
extern "C" bool launch_w4a8_codebook_mma(
    const void* xq, const void* weight, const void* decode_lut,
    const void* s_channel, const void* xs, const void* bias, void* workspace,
    void* counters, void* out, int64_t M, int64_t N, int64_t K, int64_t G,
    int64_t split_k, int64_t warps_per_block, int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (M > 8 || decode_lut == nullptr || N > std::numeric_limits<int>::max()
            || K > std::numeric_limits<int>::max() || K % 32 != 0
            || G != 16 || K % G != 0 || split_k <= 0
            || K % (split_k * 32) != 0
            || (warps_per_block != 1 && warps_per_block != 2
                && warps_per_block != 4 && warps_per_block != 8)) {
        return false;
    }
    int device = 0;
    int compute_capability_major = 0;
    if (cudaGetDevice(&device) != cudaSuccess
            || cudaDeviceGetAttribute(
                &compute_capability_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
            || compute_capability_major < 8) {
        return false;
    }
    // split_k from the caller describes the weight packing (PackRows records per
    // split); the streaming kernel picks its own rows per warp for occupancy.
    const int pack_rows = static_cast<int>(K) / static_cast<int>(split_k * 32);
    const int rows = w4a8_stream_rows(N, K, pack_rows);
    const bool streamed = rows != 0;
    const int k_rows = static_cast<int>(K) / 32;
    auto launch = [&]<int WarpsPerBlock>() {
        constexpr int OutputsPerBlock = WarpsPerBlock * 16;
        const dim3 grid(
            static_cast<unsigned int>((N + OutputsPerBlock - 1) / OutputsPerBlock),
            static_cast<unsigned int>(streamed ? k_rows / rows : split_k));
        if (streamed) {
            DISPATCH_FP_DTYPE(out_dtype_code, OutputT, [&] {
                auto launch_stream = [&]<int PackRows>() {
                    w4a8_codebook_mma_stream_kernel<WarpsPerBlock, PackRows, OutputT>
                        <<<grid, WarpsPerBlock * 32, w4a8_stream_smem_bytes(WarpsPerBlock, rows), stream>>>(
                        static_cast<const int8_t*>(xq),
                        static_cast<const int8_t*>(weight),
                        static_cast<const int8_t*>(decode_lut),
                        static_cast<const float*>(s_channel),
                        static_cast<const float*>(xs),
                        static_cast<const float*>(bias),
                        static_cast<int*>(workspace),
                        static_cast<int*>(counters),
                        static_cast<OutputT*>(out),
                        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), rows);
                };
                if (pack_rows == 8) launch_stream.template operator()<8>();
                else launch_stream.template operator()<16>();
            });
        } else {
            w4a8_codebook_mma_kernel<WarpsPerBlock>
                <<<grid, WarpsPerBlock * 32, 0, stream>>>(
                static_cast<const int8_t*>(xq),
                static_cast<const int8_t*>(weight),
                static_cast<const int8_t*>(decode_lut),
                static_cast<int*>(workspace),
                static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(K), static_cast<int>(G), static_cast<int>(split_k));
        }
    };
    switch (warps_per_block) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 4: launch.template operator()<4>(); break;
        case 8: launch.template operator()<8>(); break;
    }
    if (!streamed) {
        constexpr int EpilogueThreads = 256;
        const int epilogue_blocks = static_cast<int>((M * N + EpilogueThreads - 1) / EpilogueThreads);
        DISPATCH_FP_DTYPE(out_dtype_code, OutputT, [&] {
            w4a8_codebook_mma_epilogue<OutputT><<<epilogue_blocks, EpilogueThreads, 0, stream>>>(
                static_cast<const int*>(workspace),
                static_cast<const float*>(s_channel), static_cast<const float*>(xs),
                static_cast<const float*>(bias), static_cast<OutputT*>(out),
                static_cast<int>(M), static_cast<int>(N));
        });
        cudaMemsetAsync(workspace, 0, M * N * sizeof(int), stream);
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("W4A8 packed MMA failed: ") + cudaGetErrorString(error));
    }
    return true;
}


// W4A8 requantize in one launch: rotated weight -> packed int4 + fp8 s_rel + f32
// s_channel (codebook assign + 2 ALS scale iters + per-channel scale + optional
// stochastic rounding + pack). Reads the already-rotated weight in its native
// dtype -- same values the eager path sees. group_size fixed at 16.
// One block per row; each thread owns whole 16-wide groups.
namespace {

__device__ __forceinline__ float rq_to_float(float v) { return v; }
__device__ __forceinline__ float rq_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float rq_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ uint32_t rq_pcg(uint32_t x) {
    x = x * 747796405u + 2891336453u;
    uint32_t w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}
// uniform in [0,1) keyed by a global element index + seed
__device__ __forceinline__ float rq_uniform(int64_t idx, uint64_t seed) {
    uint32_t h = rq_pcg(static_cast<uint32_t>(idx) ^ static_cast<uint32_t>(seed)
                        ^ static_cast<uint32_t>(seed >> 32));
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float rq_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
// nearest index in cb[16] (lowest index on tie); cb sorted ascending
__device__ __forceinline__ int rq_nearest(float x, const float* cb) {
    int best = 0; float bd = fabsf(x - cb[0]);
    #pragma unroll
    for (int j = 1; j < 16; ++j) { float d = fabsf(x - cb[j]); if (d < bd) { bd = d; best = j; } }
    return best;
}

template <typename InputType, bool STOCHASTIC>
__global__ void quantize_w4a8_convrot_kernel(
    const InputType* __restrict__ rotated,  // [N, K]
    const float* __restrict__ codebook,     // [16]
    int8_t* __restrict__ packed,            // [N, K/2]
    uint8_t* __restrict__ s_rel,            // [N, K/16] e4m3 bits
    float* __restrict__ s_channel,          // [N]
    int K, uint64_t seed)
{
    constexpr int G = 16;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int groups = K / G;
    const int64_t row_off = static_cast<int64_t>(row) * K;

    __shared__ float cb[16];
    __shared__ float warp_max[32];
    extern __shared__ float gscale[];  // [groups]
    if (tid < 16) cb[tid] = codebook[tid];
    __syncthreads();

    // --- Phase 1: per-group ALS group scale; accumulate row shifted-amax ---
    float thr_amax = 0.0f;
    for (int g = tid; g < groups; g += nthreads) {
        const int64_t base = row_off + static_cast<int64_t>(g) * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) { w[i] = rq_to_float(rotated[base + i]); amax = fmaxf(amax, fabsf(w[i])); }
        float gs = fmaxf(amax, 1e-8f);
        int idx[G];
        #pragma unroll
        for (int i = 0; i < G; ++i) idx[i] = rq_nearest(w[i] / gs, cb);
        #pragma unroll
        for (int it = 0; it < 2; ++it) {       // matches eager _ALS_ITERS
            float num = 0.0f, den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) { float c = cb[idx[i]]; num += w[i] * c; den += c * c; }
            gs = fmaxf(num / fmaxf(den, 1e-8f), 1e-8f);
            #pragma unroll
            for (int i = 0; i < G; ++i) idx[i] = rq_nearest(w[i] / gs, cb);
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i) thr_amax = fmaxf(thr_amax, fabsf(cb[idx[i]] * gs));
    }

    // --- block reduce thr_amax -> s_channel = row_amax / 127 ---
    float wm = rq_warp_max(thr_amax);
    const int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) warp_max[wid] = wm;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = (nthreads + 31) >> 5;
        float t = (lane < nwarps) ? warp_max[lane] : 0.0f;
        t = rq_warp_max(t);
        if (lane == 0) warp_max[0] = t;
    }
    __syncthreads();
    const float sc = fmaxf(warp_max[0] / 127.0f, 1e-8f);
    if (tid == 0) s_channel[row] = sc;

    // --- Phase 2: s_rel (fp8) -> int8 levels -> assign (+SR) -> pack ---
    const int64_t prow_off = static_cast<int64_t>(row) * (K / 2);
    const int64_t srow_off = static_cast<int64_t>(row) * groups;
    for (int g = tid; g < groups; g += nthreads) {
        const float gs = gscale[g];
        const float srel_f = gs / sc;
        const uint8_t srel_bits = __nv_cvt_float_to_fp8(srel_f, __NV_SATFINITE, __NV_E4M3);
        s_rel[srow_off + g] = srel_bits;
        const float srel_r = __half2float(__nv_cvt_fp8_to_halfraw(srel_bits, __NV_E4M3));
        float lv[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) lv[j] = fminf(127.0f, fmaxf(-127.0f, nearbyintf(cb[j] * srel_r)));
        const int64_t base = row_off + static_cast<int64_t>(g) * G;
        int u[G];
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            const float t = rq_to_float(rotated[base + i]) / sc;
            int a;
            if constexpr (STOCHASTIC) {
                int lo = 0;
                #pragma unroll
                for (int j = 0; j < 16; ++j) lo += (lv[j] <= t);
                lo = min(max(lo - 1, 0), 14);
                const float thr = lv[lo] + rq_uniform(base + i, seed) * (lv[lo + 1] - lv[lo]);
                a = min(lo + (t > thr ? 1 : 0), 15);
            } else {
                a = rq_nearest(t, lv);
            }
            u[i] = a;
        }
        const int64_t base_p = prow_off + static_cast<int64_t>(g) * (G / 2);
        #pragma unroll
        for (int p = 0; p < G / 2; ++p)
            packed[base_p + p] = static_cast<int8_t>((u[2 * p] & 0xF) | ((u[2 * p + 1] & 0xF) << 4));
    }
}

}  // namespace

// rotated: [N,K] in in_dtype (0=fp32,1=fp16,2=bf16); s_rel: [N,K/16] e4m3 bits (uint8).
// Returns false (caller must fall back / raise) if the group-scale shared memory won't fit
// or the launch is rejected, so uninitialized outputs are never mistaken for a result.
extern "C" bool launch_quantize_w4a8_convrot(
    const void* rotated, const void* codebook, void* packed, void* s_rel, void* s_channel,
    int64_t N, int64_t K, int in_dtype_code, bool stochastic, uint64_t seed, cudaStream_t stream)
{
    const int threads = 256;
    const size_t shmem = static_cast<size_t>(K / 16) * sizeof(float);
    // Static shared is cb[16] + warp_max[32] = 192 bytes; bail if static+dynamic won't fit.
    int dev = 0, max_shmem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlock, dev) != cudaSuccess)
        return false;
    if (shmem + 192 > static_cast<size_t>(max_shmem)) return false;
    dim3 grid(static_cast<unsigned>(N));
#define RQ_LAUNCH(IT)                                                                              \
    do {                                                                                           \
        if (stochastic)                                                                            \
            quantize_w4a8_convrot_kernel<IT, true><<<grid, threads, shmem, stream>>>(              \
                static_cast<const IT*>(rotated), static_cast<const float*>(codebook),              \
                static_cast<int8_t*>(packed), static_cast<uint8_t*>(s_rel),                        \
                static_cast<float*>(s_channel), static_cast<int>(K), seed);                        \
        else                                                                                       \
            quantize_w4a8_convrot_kernel<IT, false><<<grid, threads, shmem, stream>>>(             \
                static_cast<const IT*>(rotated), static_cast<const float*>(codebook),              \
                static_cast<int8_t*>(packed), static_cast<uint8_t*>(s_rel),                        \
                static_cast<float*>(s_channel), static_cast<int>(K), 0);                           \
    } while (0)
    if (in_dtype_code == 0) RQ_LAUNCH(float);
    else if (in_dtype_code == 1) RQ_LAUNCH(__half);
    else RQ_LAUNCH(__nv_bfloat16);
#undef RQ_LAUNCH
    return cudaGetLastError() == cudaSuccess;
}

// Fused W4A8 GEMV for decode (M<=8): dequantize int4+codebook in registers and
// dp4a against the int8 activation in one pass — no int8 workspace round-trip.
// Bit-exact with the chunked path: same __float2int_rn(cb[c]*s_rel) int8 grid,
// same acc*xs*s_channel(+bias) epilogue. Requires G>=16 and G%16==0 so one
// 16-col vec never spans two groups.
namespace {

constexpr int kGemvMaxM = 8;  // sizes acc[] below and gates the launcher

template <int WARPS_PER_BLOCK, typename OutT>
__global__ void w4a8_codebook_gemv_kernel(
    const int8_t* __restrict__ xq,        // (M, K) int8 rotated+quantized activation
    const int8_t* __restrict__ qw,        // (N, K/2) packed uint4
    const uint8_t* __restrict__ s_rel,    // (N, K/G) e4m3 raw
    const float* __restrict__ codebook,   // 16 floats or nullptr
    const float* __restrict__ s_channel,  // (N)
    const float* __restrict__ xs,         // (M)
    const OutT* __restrict__ bias,        // (N) in the output dtype, or nullptr
    OutT* __restrict__ out,               // (M, N)
    int M, int N, int K, int G)
{
    __shared__ float cb[16];
    if (threadIdx.x < 16)
        cb[threadIdx.x] = codebook ? codebook[threadIdx.x] : (static_cast<float>(threadIdx.x) - 8.0f);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    if (n >= N)
        return;

    const int Khalf = K >> 1;
    const int nvec = Khalf >> 3;                  // 16-col vecs per row
    const int nG = K / G;
    const int8_t* __restrict__ wrow = qw + static_cast<int64_t>(n) * Khalf;
    const uint8_t* __restrict__ srow = s_rel + static_cast<int64_t>(n) * nG;

    // one weight pass shared across all M rows (M <= kGemvMaxM)
    int acc[kGemvMaxM] = {};
    for (int v = lane; v < nvec; v += 32) {
        const uint2 pk = *reinterpret_cast<const uint2*>(wrow + v * 8);
        const int k0 = v * 16;
        const float s = load_scale<uint8_t>(srow[k0 / G]);
        char4 w4[4];
        dequant16_int4_to_int8(pk, cb, s, s, s, s, G, w4);
        const int kw = k0 >> 2;
        for (int m = 0; m < M; ++m) {
            const int* __restrict__ x4 = reinterpret_cast<const int*>(xq + static_cast<int64_t>(m) * K);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[m] = __dp4a(x4[kw + j], *reinterpret_cast<const int*>(&w4[j]), acc[m]);
        }
    }

    for (int m = 0; m < M; ++m) {
        int a = acc[m];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) {
            float value = static_cast<float>(a) * xs[m] * s_channel[n];
            if (bias)
                value += static_cast<float>(bias[n]);
            out[static_cast<int64_t>(m) * N + n] = comfy::from_float<OutT>(value);
        }
    }
}

}  // namespace

extern "C" bool launch_w4a8_codebook_gemv(
    const void* xq, const void* weight, const void* s_rel, const void* codebook,
    const void* s_channel, const void* xs, const void* bias, void* out,
    int64_t M, int64_t N, int64_t K, int64_t G,
    int out_dtype_code, cudaStream_t stream)
{
    if (M < 1 || M > kGemvMaxM)
        return false;
    constexpr int kWarps = 4;
    dim3 block(kWarps * 32);
    dim3 grid((N + kWarps - 1) / kWarps);
#define GEMV_LAUNCH(OT)                                                                            \
    w4a8_codebook_gemv_kernel<kWarps, OT><<<grid, block, 0, stream>>>(                             \
        static_cast<const int8_t*>(xq), static_cast<const int8_t*>(weight),                        \
        static_cast<const uint8_t*>(s_rel), static_cast<const float*>(codebook),                   \
        static_cast<const float*>(s_channel), static_cast<const float*>(xs),                       \
        static_cast<const OT*>(bias), static_cast<OT*>(out),                                       \
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), static_cast<int>(G))
    if (out_dtype_code == 0) GEMV_LAUNCH(float);
    else if (out_dtype_code == 1) GEMV_LAUNCH(__half);
    else GEMV_LAUNCH(__nv_bfloat16);
#undef GEMV_LAUNCH
    return cudaGetLastError() == cudaSuccess;
}
