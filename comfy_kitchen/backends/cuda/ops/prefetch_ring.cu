#include <cuda_runtime.h>

#include <cstdint>

#include "../prefetch_ring.h"

namespace {

// Issuer geometry, measured on RTX 5090 / CUDA 13.0 (pfbench/pf_rate.cu,
// contend4_bench.py): one cp.async.bulk.prefetch.L2 issuer thread sustains
// ~170 GB/s and self-throttles, 10 of them reach the DRAM ceiling (1.64-1.71
// TB/s) with every line landing; from 14 on, requests are dropped. Prefetch
// requests return no data to the SM, which is what makes them cheap for a
// concurrently running L2-hitting GEMM: an ld-based issuer at the same rate
// slows such a GEMM about twice as much (contend3_bench.py).
constexpr int kIssuers = PREFETCH_RING_ISSUERS;
#ifndef PREFETCH_RING_ISSUER_CARVEOUT
#define PREFETCH_RING_ISSUER_CARVEOUT cudaSharedmemCarveoutMaxShared
#endif
constexpr int kIssuerThreads = 32;
constexpr int kIssueBatch = 8;   // chunks issued per consumed-snapshot
// Experiment: aggregate issue-rate cap (GB/s) applied once the issuer leads demand by
// PREFETCH_RING_PACE_LEAD_MIB. Bulk-prefetch fills above ~1.5 TB/s slow every kernel sharing
// the L2 by 40-100% (fill_penalty.py); below 1.4 TB/s the penalty is under 15%.
#ifndef PREFETCH_RING_PACE_GBPS
#define PREFETCH_RING_PACE_GBPS 0
#endif
#ifndef PREFETCH_RING_PACE_LEAD_MIB
#define PREFETCH_RING_PACE_LEAD_MIB 0
#endif
constexpr uint64_t kPaceLead = uint64_t(PREFETCH_RING_PACE_LEAD_MIB) << 20;
PrefetchRingState* g_states[16] = {};
cudaStream_t g_issue_streams[16] = {};
cudaEvent_t g_start_events[16] = {};

__device__ uint64_t region_total(const PrefetchRegion* regions, int count) {
    uint64_t total = 0;
    for (int i = 0; i < count; ++i) total += regions[i].bytes;
    return total;
}

__global__ void configure_prefetch_ring_kernel(
    PrefetchRingState* ring, const PrefetchRegion* regions, int count,
    uint64_t lookahead, uint32_t chunk, uint32_t credits) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    ring->regions = regions;
    ring->count = count;
    ring->total = region_total(regions, count);
    ring->lookahead = lookahead;
    ring->chunk = chunk;
    ring->credits = credits;
    ring->stalled = 0;
    ring->consumed = 0;
    ring->next_batch = 0;
    ring->touched = ring->skipped = ring->waited_ns = 0;
    ring->arrived = 0;
    for (int i = 0; i <= kIssuers; ++i) ring->distinct_hist[i] = 0;
    ring->enabled = count > 0 && ring->total > 0;
}

__global__ void disable_prefetch_ring_kernel(PrefetchRingState* ring) {
    if (blockIdx.x == 0 && threadIdx.x == 0) ring->enabled = 0;
}

// Runs on the consumer stream at step start: every consumer of the previous
// step (including W4A8 kernels outside the ring scope) has finished, so the
// step's byte count restarts at zero. Re-arms a ring that was disabled
// mid-step (a host sync must stop the issuer first; see prefetch_ring_disable).
// Region byte counts may change between steps (KV rows grow), so the step
// total is summed here from the descriptors the host updated on `stream`.
__global__ void reset_prefetch_ring_kernel(PrefetchRingState* ring) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ring->total = region_total(ring->regions, ring->count);
        ring->consumed = 0;
        ring->next_batch = 0;
        ring->arrived = 0;
        ring->trace_n = 0;
        ring->enabled = ring->count > 0 && ring->total > 0;
    }
}

__global__ void set_prefetch_ring_trace_kernel(PrefetchRingState* ring, uint64_t* trace, uint32_t cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ring->trace = trace;
        ring->trace_cap = cap;
        ring->trace_n = 0;
    }
}

__device__ __forceinline__ uint64_t global_timer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void trace_record(PrefetchRingState* ring, uint64_t* trace, uint32_t cap,
                                             uint64_t consumed, uint64_t cursor, uint32_t event) {
    if (trace == nullptr) return;
    const uint32_t i = atomicAdd(&ring->trace_n, 1u);
    if (i >= cap) return;
    uint64_t* r = trace + static_cast<size_t>(i) * 4;
    r[0] = global_timer();
    r[1] = consumed;
    r[2] = cursor;
    r[3] = blockIdx.x | (static_cast<uint64_t>(event) << 8);
}

// Cursor into the concatenated region list; advancing by n bytes walks regions.
// The current region is cached in registers: the issue loop must not take a
// dependent global load per chunk (see the issuer geometry note).
struct RingCursor {
    const PrefetchRegion* regions;
    int count;
    int index;
    uint64_t offset;
    const unsigned char* base;
    uint64_t bytes;

    __device__ void load() {
        const PrefetchRegion r = regions[index];
        base = r.base;
        bytes = r.bytes;
    }
    __device__ void next_region() {
        offset = 0;
        if (++index == count) index = 0;
        load();
    }
    __device__ void advance(uint64_t n) {
        while (n != 0) {
            const uint64_t avail = bytes - offset;
            const uint64_t step = n < avail ? n : avail;
            n -= step;
            offset += step;
            if (offset == bytes) next_region();
        }
    }
    // Request [cursor, cursor + n) into L2, one bulk prefetch per region piece.
    __device__ void prefetch(uint64_t n) {
        while (n != 0) {
            const uint64_t avail = bytes - offset;
            const uint64_t step = n < avail ? n : avail;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                         :: "l"(base + offset), "r"(static_cast<unsigned>(step)) : "memory");
            n -= step;
            offset += step;
            if (offset == bytes) next_region();
        }
    }
};

// One thread per CTA. The step's byte stream is cut into batches of kIssueBatch
// chunks that the CTAs claim in order from `next_batch`, so the prefetched
// frontier is one contiguous run behind the claim counter no matter how
// unevenly the CTAs progress. (Progress is uneven: bulk prefetch issue is
// throttled per TPC at ~370 GB/s, so a CTA sharing its TPC with another issuer
// or with GEMM CTAs runs at ~160 GB/s while one alone on its TPC runs at ~300.
// With a static interleave the frontier moved at 10x the slowest CTA and the
// fast ones idled at the window edge.) A chunk is requested once it lies
// within `lookahead` of the consumed position; chunks demand has already
// consumed are skipped. The kernel ends after requesting `lookahead` bytes
// past the end of the step (the wrap-around start of the next one).
//
// The loop decides from a snapshot of the racy counters, and a batch claim,
// that were loaded while the previous batch issued, so the fast path has no
// load latency in it; a stale snapshot only delays a skip or a window advance
// by one batch. The counters are polled synchronously only while the window
// is exhausted.

__global__ void __launch_bounds__(kIssuerThreads) prefetch_ring_issuer_kernel(PrefetchRingState* ring) {
    if (threadIdx.x != 0) return;
    {
        unsigned sm;
        asm("mov.u32 %0, %%smid;" : "=r"(sm));
        ring->smid[blockIdx.x] = sm;
        __threadfence();
        if (atomicAdd(&ring->arrived, 1u) == gridDim.x - 1) {   // last CTA to start: count distinct SMs
            int distinct = 0;
            for (int i = 0; i < static_cast<int>(gridDim.x); ++i) {
                const unsigned si = *reinterpret_cast<volatile uint32_t*>(&ring->smid[i]);
                bool seen = false;
                for (int j = 0; j < i; ++j) seen |= *reinterpret_cast<volatile uint32_t*>(&ring->smid[j]) == si;
                distinct += !seen;
            }
            atomicAdd(&ring->distinct_hist[distinct], 1u);
        }
    }
    const uint64_t end = ring->total;
    const uint64_t lookahead = ring->lookahead;
    const uint64_t chunk = ring->chunk;
    const uint64_t batch = chunk * kIssueBatch;
    volatile uint64_t* consumed_p = &ring->consumed;
    volatile int* enabled_p = &ring->enabled;
    unsigned long long* next_batch_p = reinterpret_cast<unsigned long long*>(&ring->next_batch);
    uint64_t* const trace = ring->trace;
    const uint32_t trace_cap = ring->trace_cap;

    uint64_t cursor = atomicAdd(next_batch_p, 1ull) * batch;   // start of this CTA's batch
    RingCursor pos{ring->regions, ring->count, 0, 0, nullptr, 0};
    pos.load();
    pos.advance(cursor);

    uint64_t touched = 0, skipped = 0, waited = 0;
    // ns per issue batch for this CTA's share of the aggregate cap; 0 = unpaced
    const uint64_t pace_ns = PREFETCH_RING_PACE_GBPS ? batch * kIssuers / uint64_t(PREFETCH_RING_PACE_GBPS) : 0;
    uint64_t next_ok = 0;
    uint64_t consumed = *consumed_p;
    int enabled = *enabled_p;
    while (cursor < end + lookahead && enabled) {
        // Claim the next batch and refresh the snapshot first; both are only
        // read after this batch issued, so their L2 round trips (which grow
        // under GEMM hit traffic) overlap the issues instead of gating them:
        // one dependent load per chunk capped the issuer at ~1.2 TB/s during
        // GEMMs.
        const uint64_t next_cursor = atomicAdd(next_batch_p, 1ull) * batch;
        const uint64_t next_consumed = *consumed_p;
        const int next_enabled = *enabled_p;
        const uint64_t batch_end = cursor + batch < end + lookahead ? cursor + batch : end + lookahead;
        // chunks demand already read are skipped; the rest are issued once inside the window
        const uint64_t from = cursor > consumed ? cursor : (consumed / chunk * chunk < batch_end ? consumed / chunk * chunk : batch_end);
        skipped += from - cursor;
        trace_record(ring, trace, trace_cap, consumed, cursor,
                     from == batch_end ? PREFETCH_RING_TRACE_SKIP : PREFETCH_RING_TRACE_ISSUE);
        pos.advance(from - cursor);
        if (pace_ns != 0) {
            // token bucket: paced only while comfortably ahead of demand, so an empty
            // ring still fills at full speed
            const uint64_t now = global_timer();
            if (from > consumed + kPaceLead) {
                uint64_t t = now;
                while (t < next_ok) {
                    __nanosleep(128);
                    t = global_timer();
                }
                next_ok = (next_ok > now ? next_ok : now) + pace_ns;
            } else {
                next_ok = now + pace_ns;
            }
        }
        for (uint64_t c = from; c < batch_end; c += chunk) {
            if (c >= consumed + lookahead) {
                const uint64_t t0 = global_timer();
                do {
                    __nanosleep(256);
                    waited += 256;
                    consumed = *consumed_p;
                    enabled = *enabled_p;
                    if (!enabled) break;
                    if (global_timer() - t0 > 100000000ull) {   // 100 ms without consumption
                        atomicAdd(&ring->stalled, 1u);
                        enabled = 0;
                        break;
                    }
                } while (c >= consumed + lookahead);
                trace_record(ring, trace, trace_cap, consumed, c, PREFETCH_RING_TRACE_WAIT);
                if (!enabled) break;
            }
            const uint64_t n = c + chunk < batch_end ? chunk : batch_end - c;
            pos.prefetch(n);
            touched += n;
        }
        if (!enabled) break;
        pos.advance(next_cursor - batch_end);
        cursor = next_cursor;
        consumed = next_consumed > consumed ? next_consumed : consumed;
        enabled = next_enabled;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&ring->touched), touched);
    atomicAdd(reinterpret_cast<unsigned long long*>(&ring->skipped), skipped);
    atomicAdd(reinterpret_cast<unsigned long long*>(&ring->waited_ns), waited);
}


PrefetchRingState* current_state(int* device_out) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16)
        return nullptr;
    if (g_states[device] == nullptr) {
        if (cudaMalloc(&g_states[device], sizeof(PrefetchRingState)) != cudaSuccess)
            return nullptr;
        cudaMemset(g_states[device], 0, sizeof(PrefetchRingState));
        cudaStreamCreateWithFlags(&g_issue_streams[device], cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&g_start_events[device], cudaEventDisableTiming);
        // The issuer CTAs are resident for the whole step. An SM only hosts CTAs
        // that agree on its L1/shared carveout, and a kernel with no shared
        // memory would be configured for maximum L1, keeping the streamed GEMM
        // (~46 KB dynamic shared per CTA) off those SMs for the whole step.
        // Ask for the GEMM's configuration so the SMs stay shared.
        cudaFuncSetAttribute(prefetch_ring_issuer_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             PREFETCH_RING_ISSUER_CARVEOUT);
        set_w4a8_prefetch_ring_state(g_states[device]);
        set_int8_prefetch_ring_state(g_states[device]);
        set_flash_prefetch_ring_state(g_states[device]);
        set_gated_delta_prefetch_ring_state(g_states[device]);
    }
    if (device_out) *device_out = device;
    return g_states[device];
}

} // namespace

bool prefetch_ring_is_available() {
    int device = 0;
    int major = 0;
    return cudaGetDevice(&device) == cudaSuccess
        && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess
        && major >= 9;
}

extern "C" void launch_prefetch_ring_configure(
    const uint64_t* regions, int count, uint64_t lookahead, uint32_t chunk, uint32_t credits,
    cudaStream_t stream) {
    PrefetchRingState* state = current_state(nullptr);
    if (state == nullptr) return;
    configure_prefetch_ring_kernel<<<1, 1, 0, stream>>>(
        state, reinterpret_cast<const PrefetchRegion*>(regions), count, lookahead, chunk, credits);
}

extern "C" void launch_prefetch_ring_disable(cudaStream_t stream) {
    PrefetchRingState* state = current_state(nullptr);
    if (state != nullptr) disable_prefetch_ring_kernel<<<1, 1, 0, stream>>>(state);
}

// Start the issuer for one step on the ring's side stream, ordered after the
// work already enqueued on `stream` (configure / previous step) and the
// counter reset.
extern "C" void launch_prefetch_ring_start(cudaStream_t stream) {
    int device = 0;
    PrefetchRingState* state = current_state(&device);
    if (state == nullptr) return;
    reset_prefetch_ring_kernel<<<1, 1, 0, stream>>>(state);
    cudaEventRecord(g_start_events[device], stream);
    cudaStreamWaitEvent(g_issue_streams[device], g_start_events[device], 0);
    prefetch_ring_issuer_kernel<<<kIssuers, kIssuerThreads, 0, g_issue_streams[device]>>>(state);
}

extern "C" void launch_prefetch_ring_set_trace(uint64_t* trace, uint32_t cap, cudaStream_t stream) {
    PrefetchRingState* state = current_state(nullptr);
    if (state != nullptr) set_prefetch_ring_trace_kernel<<<1, 1, 0, stream>>>(state, trace, cap);
}

// Diagnostic: synchronous copy of the state counters.
extern "C" void prefetch_ring_read_stats(PrefetchRingState* host) {
    PrefetchRingState* state = current_state(nullptr);
    *host = PrefetchRingState{};
    if (state != nullptr) cudaMemcpy(host, state, sizeof(*host), cudaMemcpyDeviceToHost);
}
