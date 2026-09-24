#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// L2 prefetch ring for decode weight sweeps.
//
// The host records the byte ranges one decode step reads, in the order the
// consumer kernels read them (a scatter-gather work list; for the streamed W4A8
// GEMM that is chunk order, not address order). Consumers report `consumed`,
// the bytes of that stream whose loads have completed. An issuer kernel on a
// side stream walks the list and requests [consumed, consumed + lookahead)
// into L2 with cp.async.bulk.prefetch.L2 (no data return), so the L2 footprint
// of prefetched-but-unread weights never exceeds `lookahead` and DRAM stays
// busy while the consumer stream runs kernels that do not read weights. It
// exits when the step is consumed. Ranges must be 16-byte aligned in base and
// size. Counters are updated without synchronization by design: the consumer
// credits with fire-and-forget atomics and the issuer reads snapshots; the
// only consequence of a stale read is a chunk prefetched late or twice.
#ifndef PREFETCH_RING_ISSUERS
#define PREFETCH_RING_ISSUERS 10
#endif

struct PrefetchRegion {
    const unsigned char* base;
    uint64_t bytes;
};

struct PrefetchRingState {
    const PrefetchRegion* regions;
    int count;
    int enabled;
    uint64_t total;       // sum of region bytes (one step)
    uint64_t lookahead;   // bytes kept in flight ahead of `consumed`
    uint32_t chunk;       // bytes per prefetch request
    uint32_t stalled;     // issuer CTAs that gave up waiting for consumption (diagnostic)
    uint64_t consumed;    // bytes whose demand loads completed this step
    // diagnostics, cumulative over steps (read with prefetch_ring_read_stats)
    uint64_t touched;     // bytes the issuer requested
    uint64_t skipped;     // bytes the issuer skipped because demand got there first
    uint64_t waited_ns;   // CTA-nanoseconds spent throttled at consumed + lookahead
    uint32_t smid[PREFETCH_RING_ISSUERS];    // SM each issuer CTA ran on in the last step (placement diagnostic)
    uint32_t arrived;     // issuer CTAs started this step
    uint32_t distinct_hist[PREFETCH_RING_ISSUERS + 1];   // steps by number of distinct SMs the issuer CTAs landed on
};

#ifdef __CUDACC__
static __device__ __forceinline__ void prefetch_ring_consume_device(
    PrefetchRingState* ring, uint64_t bytes) {
    if (ring != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(&ring->consumed),
                  static_cast<unsigned long long>(bytes));
}
#endif

bool prefetch_ring_is_available();

extern "C" void launch_prefetch_ring_configure(
    const uint64_t* regions, int count, uint64_t lookahead, uint32_t chunk,
    cudaStream_t stream);
extern "C" void launch_prefetch_ring_disable(cudaStream_t stream);
extern "C" void launch_prefetch_ring_start(cudaStream_t stream);
extern "C" void set_w4a8_prefetch_ring_state(PrefetchRingState* state);
extern "C" void prefetch_ring_read_stats(PrefetchRingState* host);
