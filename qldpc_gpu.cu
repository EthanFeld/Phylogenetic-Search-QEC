#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifdef _WIN32
#define QLDPC_GPU_EXPORT __declspec(dllexport)
#else
#define QLDPC_GPU_EXPORT
#endif

namespace {

using word_t = std::uint32_t;

__device__ __forceinline__ std::uint64_t next_random(std::uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

__device__ __forceinline__ bool bit_at_word(const word_t* matrix, int row,
                                            int word_index, word_t mask, int rows) {
    return (matrix[word_index * rows + row] & mask) != 0;
}

__device__ int row_weight(const word_t* matrix, int row, int rows, int limbs) {
    int out = 0;
    for (int limb = 0; limb < limbs; ++limb)
        out += __popc(matrix[limb * rows + row]);
    return out;
}

__device__ int packed_weight(const word_t* row, int limbs) {
    int out = 0;
    for (int limb = 0; limb < limbs; ++limb) out += __popc(row[limb]);
    return out;
}

__device__ bool signature_any(const word_t* signature, int limbs) {
    for (int limb = 0; limb < limbs; ++limb)
        if (signature[limb]) return true;
    return false;
}

__device__ void publish_candidate(const word_t* row, int weight,
                                  int n, int limbs, int target,
                                  int stop_on_target, int* best, int* lock,
                                  int* stop, word_t* witness) {
    if (weight <= 0) return;
    if (weight >= *best) return;
    while (atomicCAS(lock, 0, 1) != 0) {}
    if (weight < *best) {
        *best = weight;
        for (int limb = 0; limb < limbs; ++limb)
            witness[limb] = row[limb];
        __threadfence();
    }
    const bool hit = target > 0 && *best < target;
    atomicExch(lock, 0);
    if (stop_on_target && hit) atomicExch(stop, 1);
    (void)n;
}

__global__ void randomized_echelon_ris_kernel(
    const word_t* __restrict__ kernel, int rows, int limbs, int n,
    const word_t* __restrict__ dual, int dual_rows, int pair_depth,
    std::uint64_t seed, std::uint64_t trial_base, int target,
    int stop_on_target, int block_width, int rank_cap,
    int* best, int* completed, int* lock, int* stop,
    word_t* witness) {
    extern __shared__ std::uint32_t shared[];
    word_t* work = reinterpret_cast<word_t*>(shared);
    std::uint32_t* meta = reinterpret_cast<std::uint32_t*>(work + rows * limbs);
    std::uint32_t* order = meta;
    std::uint32_t* weights = order + n;
    std::uint32_t* light = weights + rows;
    const int meta_words = (n + rows + pair_depth + 1) & ~1;
    word_t* temp = reinterpret_cast<word_t*>(meta + meta_words);
    const int signature_limbs = (dual_rows + 31) / 32;
    word_t* signatures = temp + limbs;
    word_t* local_witness = signatures + rows * signature_limbs;
    const std::uint64_t trial = trial_base + static_cast<std::uint64_t>(blockIdx.x);
    __shared__ int pivot;
    __shared__ int order_index_shared;
    __shared__ int selected_col;

    if (*stop) return;
    for (int index = static_cast<int>(threadIdx.x);
         index < rows * limbs; index += static_cast<int>(blockDim.x)) {
        const int row = index / limbs;
        const int limb = index % limbs;
        work[limb * rows + row] = kernel[index];
    }
    if (threadIdx.x == 0) {
        for (int col = 0; col < n; ++col) order[col] = static_cast<std::uint32_t>(col);
        std::uint64_t state = seed ^ (trial * 0x9E3779B97F4A7C15ULL);
        if (state == 0) state = 0xD1B54A32D192ED03ULL;
        for (int col = n - 1; col > 0; --col) {
            const int other = static_cast<int>(next_random(state) % static_cast<std::uint64_t>(col + 1));
            const std::uint32_t value = order[col];
            order[col] = order[other];
            order[other] = value;
        }
    }
    __syncthreads();

    const int rank_limit = rank_cap > 0 ? min(rank_cap, rows) : rows;
    int rank = 0;
    if (threadIdx.x == 0) order_index_shared = 0;
    __syncthreads();
    while (rank < rank_limit) {
        if (threadIdx.x == 0) {
            pivot = rows;
            if (order_index_shared < n) {
                selected_col = static_cast<int>(order[order_index_shared++]);
                const int selected_word = selected_col >> 5;
                const word_t selected_mask = word_t(1) << (selected_col & 31);
                for (int row = rank; row < rows; ++row) {
                    if (bit_at_word(work, row, selected_word, selected_mask, rows)) {
                        pivot = row;
                        break;
                    }
                }
            }
        }
        __syncthreads();
        if (pivot == rows) {
            if (order_index_shared >= n) break;
            continue;
        }
        if (threadIdx.x == 0) {
            if (pivot != rank) {
                #pragma unroll 4
                for (int limb = 0; limb < limbs; ++limb) {
                    const word_t value = work[limb * rows + rank];
                    work[limb * rows + rank] = work[limb * rows + pivot];
                    work[limb * rows + pivot] = value;
                }
            }
            weights[rank] = static_cast<std::uint32_t>(selected_col);
        }
        __syncthreads();
        for (int row = rank + static_cast<int>(threadIdx.x);
             row < rows; row += static_cast<int>(blockDim.x)) {
            const int selected_word = selected_col >> 5;
            const word_t selected_mask = word_t(1) << (selected_col & 31);
            if (row == rank || !bit_at_word(work, row, selected_word,
                                            selected_mask, rows)) continue;
            #pragma unroll 4
            for (int limb = 0; limb < limbs; ++limb)
                work[limb * rows + row] ^= work[limb * rows + rank];
        }
        __syncthreads();
        ++rank;
    }

    // Forward echelon form is sufficient for RIS: every retained row is a
    // valid kernel vector, and pair-shell sums remain valid kernel vectors.

    if (dual_rows > 0) {
        for (int index = static_cast<int>(threadIdx.x);
             index < rank * signature_limbs; index += static_cast<int>(blockDim.x)) {
            const int row = index / signature_limbs;
            const int signature_limb = index % signature_limbs;
            word_t signature = 0;
            const int first_detector = signature_limb * 32;
            const int last_detector = min(first_detector + 32, dual_rows);
            for (int detector_index = first_detector;
                 detector_index < last_detector; ++detector_index) {
                int parity = 0;
                for (int limb = 0; limb < limbs; ++limb)
                    parity ^= (__popc(work[limb * rows + row] &
                                      dual[detector_index * limbs + limb]) & 1);
                if (parity) signature |= 1U << (detector_index - first_detector);
            }
            signatures[index] = signature;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int keep = min(max(pair_depth, 1), rows);
        int local_best = n + 1;
        int light_count = 0;
        for (int row = 0; row < rank; ++row) {
            const int weight = row_weight(work, row, rows, limbs);
            weights[row] = static_cast<std::uint32_t>(weight);
            const bool logical = dual_rows == 0 || signature_any(
                signatures + row * signature_limbs, signature_limbs);
            if (weight > 0 && logical && weight < local_best) {
                local_best = weight;
                for (int limb = 0; limb < limbs; ++limb)
                    local_witness[limb] = work[limb * rows + row];
            }
            if (light_count < keep) {
                light[light_count++] = static_cast<std::uint32_t>(row);
            } else {
                int worst = 0;
                for (int i = 1; i < keep; ++i)
                    if (weights[light[i]] > weights[light[worst]]) worst = i;
                if (weight < static_cast<int>(weights[light[worst]]))
                    light[worst] = static_cast<std::uint32_t>(row);
            }
        }
        for (int a = 0; a + 1 < light_count; ++a) {
            for (int b = a + 1; b < light_count; ++b) {
                const int row_a = static_cast<int>(light[a]);
                const int row_b = static_cast<int>(light[b]);
                for (int limb = 0; limb < limbs; ++limb)
                    temp[limb] = work[limb * rows + row_a] ^ work[limb * rows + row_b];
                const int weight = packed_weight(temp, limbs);
                bool logical = dual_rows == 0;
                if (dual_rows > 0) {
                    for (int limb = 0; limb < signature_limbs; ++limb) {
                        if (signatures[row_a * signature_limbs + limb] ^
                            signatures[row_b * signature_limbs + limb]) {
                            logical = true;
                            break;
                        }
                    }
                }
                if (weight > 0 && logical && weight < local_best) {
                    local_best = weight;
                    for (int limb = 0; limb < limbs; ++limb)
                        local_witness[limb] = temp[limb];
                }
            }
        }
        if (local_best <= n)
            publish_candidate(local_witness, local_best, n, limbs, target,
                              stop_on_target, best, lock, stop, witness);
        atomicAdd(completed, 1);
    }
}

int check_cuda(cudaError_t error) {
    return error == cudaSuccess ? 0 : -1;
}

}  // namespace

extern "C" QLDPC_GPU_EXPORT int qldpc_gpu_classical_ris(
    const word_t* kernel, int rows, int limbs, int n,
    const word_t* dual, int dual_rows, int trials,
    std::uint64_t seed, int pair_depth, int target, int stop_on_target,
    int block_width, int rank_cap, int threads, int batch_size, int* best, int* completed,
    word_t* witness, double* milliseconds) {
    if (!kernel || !best || !completed || !witness || rows <= 0 || limbs <= 0 ||
        n <= 0 || trials < 0 || pair_depth <= 0 || dual_rows < 0 ||
        threads < 32 || threads > 1024 || (threads % 32) != 0 ||
        block_width < 1 || block_width > 8 ||
        rank_cap < 0 || rank_cap > rows ||
        batch_size < 1 || batch_size > 65536) return -2;
    word_t* d_kernel = nullptr;
    word_t* d_dual = nullptr;
    word_t* d_witness = nullptr;
    int *d_best = nullptr, *d_completed = nullptr, *d_lock = nullptr, *d_stop = nullptr;
    const std::size_t kernel_bytes = static_cast<std::size_t>(rows) * limbs * sizeof(word_t);
    const std::size_t dual_bytes = static_cast<std::size_t>(dual_rows) * limbs * sizeof(word_t);
    const std::size_t witness_bytes = static_cast<std::size_t>(limbs) * sizeof(word_t);
    int rc = 0;
    auto fail = [&]() { rc = -1; };
    if (check_cuda(cudaMalloc(&d_kernel, kernel_bytes))) fail();
    if (rc == 0 && dual_rows > 0 && check_cuda(cudaMalloc(&d_dual, dual_bytes))) fail();
    if (rc == 0 && check_cuda(cudaMalloc(&d_witness, witness_bytes))) fail();
    if (rc == 0 && check_cuda(cudaMalloc(&d_best, sizeof(int)))) fail();
    if (rc == 0 && check_cuda(cudaMalloc(&d_completed, sizeof(int)))) fail();
    if (rc == 0 && check_cuda(cudaMalloc(&d_lock, sizeof(int)))) fail();
    if (rc == 0 && check_cuda(cudaMalloc(&d_stop, sizeof(int)))) fail();
    if (rc == 0 && check_cuda(cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice))) fail();
    if (rc == 0 && dual_rows > 0 && check_cuda(cudaMemcpy(d_dual, dual, dual_bytes, cudaMemcpyHostToDevice))) fail();
    const int initial_best = n + 1;
    if (rc == 0 && check_cuda(cudaMemcpy(d_best, &initial_best, sizeof(int), cudaMemcpyHostToDevice))) fail();
    if (rc == 0 && check_cuda(cudaMemset(d_completed, 0, sizeof(int)))) fail();
    if (rc == 0 && check_cuda(cudaMemset(d_lock, 0, sizeof(int)))) fail();
    if (rc == 0 && check_cuda(cudaMemset(d_stop, 0, sizeof(int)))) fail();
    if (rc == 0 && check_cuda(cudaMemset(d_witness, 0, witness_bytes))) fail();

    cudaEvent_t start = nullptr, finish = nullptr;
    if (rc == 0 && check_cuda(cudaEventCreate(&start))) fail();
    if (rc == 0 && check_cuda(cudaEventCreate(&finish))) fail();
    const int row_stride = limbs;
    const int signature_limbs = (dual_rows + 31) / 32;
    const int meta_words = (n + rows + pair_depth + 1) & ~1;
    const int shared_bytes = static_cast<int>(
        static_cast<std::size_t>(rows) * row_stride * sizeof(word_t) +
        static_cast<std::size_t>(meta_words) * sizeof(std::uint32_t) +
         (static_cast<std::size_t>(limbs) +
         static_cast<std::size_t>(rows) * signature_limbs + limbs) * sizeof(word_t));
    if (rc == 0 && check_cuda(cudaFuncSetAttribute(
            randomized_echelon_ris_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes))) fail();
    if (rc == 0 && check_cuda(cudaEventRecord(start))) fail();
    int launched = 0;
    while (rc == 0 && launched < trials) {
        int batch = std::min(batch_size, trials - launched);
        randomized_echelon_ris_kernel<<<batch, threads, shared_bytes>>>(
            d_kernel, rows, limbs, n, d_dual, dual_rows, pair_depth, seed,
            static_cast<std::uint64_t>(launched), target, stop_on_target,
            block_width, rank_cap, d_best, d_completed, d_lock, d_stop, d_witness);
        if (check_cuda(cudaGetLastError()) || check_cuda(cudaDeviceSynchronize())) {
            rc = -1;
            break;
        }
        int stopped = 0;
        if (check_cuda(cudaMemcpy(&stopped, d_stop, sizeof(int), cudaMemcpyDeviceToHost))) {
            rc = -1;
            break;
        }
        launched += batch;
        if (stopped && stop_on_target) break;
    }
    if (rc == 0 && check_cuda(cudaEventRecord(finish))) rc = -1;
    if (rc == 0 && check_cuda(cudaEventSynchronize(finish))) rc = -1;
    if (rc == 0 && check_cuda(cudaMemcpy(best, d_best, sizeof(int), cudaMemcpyDeviceToHost))) rc = -1;
    if (rc == 0 && check_cuda(cudaMemcpy(completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost))) rc = -1;
    if (rc == 0 && check_cuda(cudaMemcpy(witness, d_witness, witness_bytes, cudaMemcpyDeviceToHost))) rc = -1;
    if (rc == 0) {
        float elapsed = 0.0f;
        if (check_cuda(cudaEventElapsedTime(&elapsed, start, finish))) rc = -1;
        *milliseconds = static_cast<double>(elapsed);
    }
    if (start) cudaEventDestroy(start);
    if (finish) cudaEventDestroy(finish);
    if (d_kernel) cudaFree(d_kernel);
    if (d_dual) cudaFree(d_dual);
    if (d_witness) cudaFree(d_witness);
    if (d_best) cudaFree(d_best);
    if (d_completed) cudaFree(d_completed);
    if (d_lock) cudaFree(d_lock);
    if (d_stop) cudaFree(d_stop);
    return rc;
}
