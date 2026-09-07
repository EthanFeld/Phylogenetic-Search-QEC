#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <algorithm>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <numeric>
#include <random>
#include <array>
#include <unordered_set>
#include <thread>
#include <vector>

#ifdef _WIN32
#define QLDPC_EXPORT extern "C" __declspec(dllexport)
#else
#define QLDPC_EXPORT extern "C"
#endif

namespace {

using Clock = std::chrono::steady_clock;

inline int popcount64(std::uint64_t x);

// Fixed-width cyclic masks cover the challenge's m <= 700 regime while
// keeping the ideal-word miner independent of Python object allocation.  The
// current Z_341 branch uses six limbs; unused high bits remain zero.
using CyclicKey = std::array<std::uint64_t, 12>;

struct CyclicKeyHash {
    std::size_t operator()(const CyclicKey& key) const noexcept {
        std::uint64_t h = 0x9E3779B97F4A7C15ULL;
        for (std::uint64_t x : key) {
            h ^= x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<std::size_t>(h ^ (h >> 32));
    }
};

inline int cyclic_popcount(const CyclicKey& key, int m) {
    int out = 0;
    const int limbs = (m + 63) / 64;
    for (int i = 0; i < limbs; ++i) out += popcount64(key[static_cast<std::size_t>(i)]);
    return out;
}

inline CyclicKey shifted_cyclic_word(const int* support, int support_len,
                                     int shift, int m) {
    CyclicKey key{};
    for (int i = 0; i < support_len; ++i) {
        // The Python bridge pads ragged support lists with -1 so a mixed
        // sparse basis can stay in one native mining call.  Padding is not a
        // polynomial exponent and must contribute no bit to the word.
        if (support[i] < 0) continue;
        int p = (support[i] + shift) % m;
        if (p < 0) p += m;
        key[static_cast<std::size_t>(p >> 6)] |= 1ULL << (p & 63);
    }
    return key;
}

inline CyclicKey xor_key(const CyclicKey& a, const CyclicKey& b) {
    CyclicKey out{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = a[i] ^ b[i];
    return out;
}

inline int popcount64(std::uint64_t x) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

struct Bits {
    std::vector<std::uint64_t> w;

    Bits() = default;
    explicit Bits(int n) : w(static_cast<std::size_t>((n + 63) / 64), 0) {}

    bool test(int bit) const {
        return ((w[static_cast<std::size_t>(bit >> 6)] >> (bit & 63)) & 1U) != 0;
    }
    void set(int bit) {
        w[static_cast<std::size_t>(bit >> 6)] |= 1ULL << (bit & 63);
    }
    bool any() const {
        for (auto x : w) if (x) return true;
        return false;
    }
    int weight() const {
        int out = 0;
        for (auto x : w) out += popcount64(x);
        return out;
    }
    int dot(const Bits& other) const {
        int parity = 0;
        const std::size_t m = std::min(w.size(), other.w.size());
        for (std::size_t i = 0; i < m; ++i) {
            parity ^= static_cast<int>(popcount64(w[i] & other.w[i]) & 1U);
        }
        return parity;
    }
    Bits& operator^=(const Bits& other) {
        for (std::size_t i = 0; i < w.size(); ++i) w[i] ^= other.w[i];
        return *this;
    }
    bool operator==(const Bits& other) const { return w == other.w; }
};

using Rows = std::vector<Bits>;

Rows dense_rows(const unsigned char* matrix, int rows, int cols) {
    Rows out;
    out.reserve(static_cast<std::size_t>(std::max(rows, 0)));
    for (int r = 0; r < rows; ++r) {
        Bits row(cols);
        for (int c = 0; c < cols; ++c) {
            if (matrix[static_cast<std::size_t>(r) * cols + c] & 1U) row.set(c);
        }
        out.push_back(std::move(row));
    }
    return out;
}

Rows rref(Rows rows, int n, const std::vector<int>& order) {
    int rank = 0;
    for (int col : order) {
        int pivot = -1;
        for (int i = rank; i < static_cast<int>(rows.size()); ++i) {
            if (rows[static_cast<std::size_t>(i)].test(col)) {
                pivot = i;
                break;
            }
        }
        if (pivot < 0) continue;
        std::swap(rows[static_cast<std::size_t>(rank)], rows[static_cast<std::size_t>(pivot)]);
        const Bits pivot_row = rows[static_cast<std::size_t>(rank)];
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            if (i != rank && rows[static_cast<std::size_t>(i)].test(col)) {
                rows[static_cast<std::size_t>(i)] ^= pivot_row;
            }
        }
        ++rank;
        if (rank == static_cast<int>(rows.size())) break;
    }
    rows.resize(static_cast<std::size_t>(rank));
    return rows;
}

std::vector<int> identity_order(int n) {
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    return order;
}

Rows kernel_basis(const unsigned char* matrix, int rows, int n) {
    Rows reduced = rref(dense_rows(matrix, rows, n), n, identity_order(n));
    std::vector<char> pivot(static_cast<std::size_t>(n), 0);
    std::vector<int> pivots;
    pivots.reserve(reduced.size());
    for (const auto& row : reduced) {
        int p = -1;
        for (int c = 0; c < n; ++c) {
            if (row.test(c)) {
                p = c;
                break;
            }
        }
        if (p >= 0) {
            pivot[static_cast<std::size_t>(p)] = 1;
            pivots.push_back(p);
        }
    }

    Rows basis;
    for (int free_col = 0; free_col < n; ++free_col) {
        if (pivot[static_cast<std::size_t>(free_col)]) continue;
        Bits x(n);
        x.set(free_col);
        for (int i = static_cast<int>(reduced.size()) - 1; i >= 0; --i) {
            if (reduced[static_cast<std::size_t>(i)].dot(x)) x.set(pivots[static_cast<std::size_t>(i)]);
        }
        basis.push_back(std::move(x));
    }
    return basis;
}

Rows representation(const int* mult, const int* inv, int s,
                   const int* support, int support_len, bool left) {
    Rows out;
    out.reserve(static_cast<std::size_t>(s));
    for (int r = 0; r < s; ++r) out.emplace_back(s);
    for (int q = 0; q < support_len; ++q) {
        const int a = support[q];
        for (int h = 0; h < s; ++h) {
            const int target = left ? mult[a * s + h] : mult[h * s + inv[a]];
            out[static_cast<std::size_t>(target)].set(h);
        }
    }
    return out;
}

bool invert_square(const Rows& matrix, int n, Rows& inverse) {
    Rows augmented;
    augmented.reserve(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        Bits row(2 * n);
        for (int c = 0; c < n; ++c) if (matrix[static_cast<std::size_t>(r)].test(c)) row.set(c);
        row.set(n + r);
        augmented.push_back(std::move(row));
    }
    const std::vector<int> order = identity_order(n);
    Rows reduced = rref(std::move(augmented), 2 * n, order);
    if (static_cast<int>(reduced.size()) != n) return false;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            if (reduced[static_cast<std::size_t>(r)].test(c) != (c == r)) return false;
        }
    }
    inverse.clear();
    inverse.reserve(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        Bits row(n);
        for (int c = 0; c < n; ++c) if (reduced[static_cast<std::size_t>(r)].test(n + c)) row.set(c);
        inverse.push_back(std::move(row));
    }
    return true;
}

int systematic_probe(const Rows& generator, int n) {
    int best = 2 * n + 1;
    for (int i = 0; i < static_cast<int>(generator.size()); ++i) {
        best = std::min(best, generator[static_cast<std::size_t>(i)].weight());
        for (int j = i + 1; j < static_cast<int>(generator.size()); ++j) {
            Bits combined = generator[static_cast<std::size_t>(i)];
            combined ^= generator[static_cast<std::size_t>(j)];
            best = std::min(best, combined.weight());
        }
    }
    return best;
}

Rows logical_basis(const unsigned char* check, int check_rows,
                   const unsigned char* stabilizer, int stabilizer_rows, int n) {
    Rows reduced = rref(dense_rows(stabilizer, stabilizer_rows, n), n, identity_order(n));
    Rows kernel = kernel_basis(check, check_rows, n);
    Rows out;
    for (auto candidate : kernel) {
        for (const auto& row : reduced) {
            int pivot = -1;
            for (int c = 0; c < n; ++c) {
                if (row.test(c)) {
                    pivot = c;
                    break;
                }
            }
            if (pivot >= 0 && candidate.test(pivot)) candidate ^= row;
        }
        if (candidate.any()) {
            out.push_back(candidate);
            reduced.push_back(std::move(candidate));
            reduced = rref(std::move(reduced), n, identity_order(n));
        }
    }
    return out;
}

struct Result {
    int best = 0;
    int trials = 0;
    bool timed_out = false;
    bool found_target = false;
    Bits witness;
};

// The challenge cap is n <= 700.  A fixed 12-limb representation removes the
// per-row heap allocations made by Bits during every randomized RREF trial.
// It is used only by the hot RIS path; the existing dynamic representation
// remains for general matrix construction and exact search.
struct FastBits {
    std::array<std::uint64_t, 12> w{};

    bool test(int bit) const {
        return ((w[static_cast<std::size_t>(bit >> 6)] >> (bit & 63)) & 1ULL) != 0;
    }
    void set(int bit) {
        w[static_cast<std::size_t>(bit >> 6)] |= 1ULL << (bit & 63);
    }
    int weight(int n) const {
        int out = 0;
        const int limbs = (n + 63) / 64;
        for (int i = 0; i < limbs; ++i) out += popcount64(w[static_cast<std::size_t>(i)]);
        return out;
    }
    bool any() const {
        for (std::uint64_t value : w) if (value != 0) return true;
        return false;
    }
    int dot(const FastBits& other, int n) const {
        int parity = 0;
        const int limbs = (n + 63) / 64;
        for (int i = 0; i < limbs; ++i)
            parity ^= static_cast<int>(popcount64(
                w[static_cast<std::size_t>(i)] & other.w[static_cast<std::size_t>(i)]) & 1U);
        return parity;
    }
    FastBits& operator^=(const FastBits& other) {
        for (std::size_t i = 0; i < w.size(); ++i) w[i] ^= other.w[i];
        return *this;
    }
};

using FastRows = std::vector<FastBits>;

FastBits to_fast(const Bits& row) {
    FastBits out;
    const std::size_t limbs = std::min<std::size_t>(out.w.size(), row.w.size());
    for (std::size_t i = 0; i < limbs; ++i) out.w[i] = row.w[i];
    return out;
}

FastRows to_fast_rows(const Rows& rows) {
    FastRows out;
    out.reserve(rows.size());
    for (const auto& row : rows) out.push_back(to_fast(row));
    return out;
}

Bits from_fast(const FastBits& row, int n) {
    Bits out(n);
    const std::size_t limbs = std::min<std::size_t>(out.w.size(), row.w.size());
    for (std::size_t i = 0; i < limbs; ++i) out.w[i] = row.w[i];
    return out;
}

void fast_rref_into(const FastRows& kernel, FastRows& rows, int n,
                    const std::vector<int>& order) {
    // Reuse row-buffer capacity across permutations instead of constructing a
    // fresh vector for every RIS trial.
    rows = kernel;
    int rank = 0;
    for (int col : order) {
        int pivot = -1;
        for (int i = rank; i < static_cast<int>(rows.size()); ++i) {
            if (rows[static_cast<std::size_t>(i)].test(col)) {
                pivot = i;
                break;
            }
        }
        if (pivot < 0) continue;
        std::swap(rows[static_cast<std::size_t>(rank)], rows[static_cast<std::size_t>(pivot)]);
        const FastBits& pivot_row = rows[static_cast<std::size_t>(rank)];
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            if (i != rank && rows[static_cast<std::size_t>(i)].test(col))
                rows[static_cast<std::size_t>(i)] ^= pivot_row;
        }
        ++rank;
        if (rank == static_cast<int>(rows.size())) break;
    }
    rows.resize(static_cast<std::size_t>(rank));
}

void fast_consider(const FastRows& rows, const FastRows& dual, int n,
                   int pair_depth, Result& result, std::vector<int>& weights,
                   std::vector<int>& light) {
    const bool require_logical = !dual.empty();
    // A negative depth requests the enhanced three-row shell.  Keeping this
    // in the existing ABI makes old callers bit-for-bit pair-only while new
    // campaigns can make a substantially stronger, still bounded proxy.
    const bool consider_triples = pair_depth < 0;
    const int requested_depth = pair_depth < 0 ? -pair_depth : pair_depth;
    weights.resize(rows.size());
    std::size_t wi = 0;
    for (const auto& row : rows) {
        const int weight = row.weight(n);
        weights[wi++] = weight;
        if (weight > 0 && weight < result.best &&
            (!require_logical || [&]() {
                for (const auto& detector : dual) if (row.dot(detector, n)) return true;
                return false;
            }())) {
            result.best = weight;
            result.witness = from_fast(row, n);
        }
    }
    if (requested_depth <= 1 || rows.size() < 2) return;
    light.resize(rows.size());
    std::iota(light.begin(), light.end(), 0);
    const std::size_t keep = std::min<std::size_t>(
        static_cast<std::size_t>(requested_depth), light.size());
    auto by_weight = [&](int a, int b) {
        const int wa = weights[static_cast<std::size_t>(a)];
        const int wb = weights[static_cast<std::size_t>(b)];
        return wa < wb || (wa == wb && a < b);
    };
    if (keep < light.size())
        std::nth_element(light.begin(), light.begin() + static_cast<std::ptrdiff_t>(keep),
                         light.end(), by_weight);
    light.resize(keep);
    std::vector<FastBits> signatures;
    if (require_logical) {
        signatures.resize(light.size());
        for (std::size_t i = 0; i < light.size(); ++i) {
            FastBits signature;
            const FastBits& row = rows[static_cast<std::size_t>(light[i])];
            for (std::size_t detector = 0; detector < dual.size(); ++detector)
                if (row.dot(dual[detector], n)) signature.set(static_cast<int>(detector));
            signatures[i] = signature;
        }
    }
    for (std::size_t a = 0; a + 1 < light.size(); ++a) {
        for (std::size_t b = a + 1; b < light.size(); ++b) {
            FastBits combined = rows[static_cast<std::size_t>(light[a])];
            combined ^= rows[static_cast<std::size_t>(light[b])];
            const int weight = combined.weight(n);
            if (weight <= 0 || weight >= result.best) continue;
            bool logical = !require_logical;
            if (require_logical) {
                FastBits signature = signatures[a];
                signature ^= signatures[b];
                logical = signature.any();
            }
            if (logical) {
                result.best = weight;
                result.witness = from_fast(combined, n);
            }
        }
    }
    if (!consider_triples || light.size() < 3) return;
    // The beam is intentionally small (campaigns use 12): C(12,3)=220
    // candidates per randomized basis, versus 276 in the old 24-row pair
    // shell.  This finds three-generator logicals before they become costly
    // false positives in deep validation.
    for (std::size_t a = 0; a + 2 < light.size(); ++a) {
        for (std::size_t b = a + 1; b + 1 < light.size(); ++b) {
            for (std::size_t c = b + 1; c < light.size(); ++c) {
                FastBits combined = rows[static_cast<std::size_t>(light[a])];
                combined ^= rows[static_cast<std::size_t>(light[b])];
                combined ^= rows[static_cast<std::size_t>(light[c])];
                const int weight = combined.weight(n);
                if (weight <= 0 || weight >= result.best) continue;
                bool logical = !require_logical;
                if (require_logical) {
                    FastBits signature = signatures[a];
                    signature ^= signatures[b];
                    signature ^= signatures[c];
                    logical = signature.any();
                }
                if (logical) {
                    result.best = weight;
                    result.witness = from_fast(combined, n);
                }
            }
        }
    }
}

Result run_ris_fast(const FastRows& kernel, const FastRows& dual, int n, int trials,
                    std::uint64_t seed, int pair_depth, double max_seconds,
                    int target, bool stop_on_target,
                    std::atomic<bool>* shared_stop = nullptr) {
    Result result;
    result.best = n + 1;
    result.witness = Bits(n);
    std::mt19937_64 rng(seed);
    std::vector<int> order = identity_order(n);
    FastRows reduced;
    reduced.reserve(kernel.size());
    std::vector<int> weights;
    std::vector<int> light;
    weights.reserve(kernel.size());
    light.reserve(kernel.size());
    const auto start = Clock::now();
    const bool has_deadline = max_seconds >= 0.0;
    const auto deadline = start + std::chrono::duration<double>(std::max(0.0, max_seconds));
    for (int trial = 0; trial < std::max(0, trials); ++trial) {
        if (shared_stop != nullptr && shared_stop->load(std::memory_order_relaxed)) break;
        std::shuffle(order.begin(), order.end(), rng);
        fast_rref_into(kernel, reduced, n, order);
        fast_consider(reduced, dual, n, pair_depth, result, weights, light);
        result.trials = trial + 1;
        if (target > 0 && result.best < target && stop_on_target) {
            result.found_target = true;
            if (shared_stop != nullptr)
                shared_stop->store(true, std::memory_order_relaxed);
            break;
        }
        if (has_deadline && ((trial + 1) & 63) == 0 && Clock::now() > deadline) {
            result.timed_out = true;
            break;
        }
    }
    return result;
}

Result run_ris_fast_parallel(const FastRows& kernel, const FastRows& dual, int n,
                             int trials, std::uint64_t seed, int pair_depth,
                             double max_seconds, int target, bool stop_on_target,
                             int requested_threads,
                             std::atomic<bool>* external_stop = nullptr) {
    const int total_trials = std::max(0, trials);
    if (total_trials == 0) {
        Result empty; empty.best = n + 1; empty.witness = Bits(n); return empty;
    }
    const int hardware = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int workers = std::max(1, std::min(total_trials,
        requested_threads > 0 ? requested_threads : hardware));
    if (workers == 1)
        return run_ris_fast(kernel, dual, n, total_trials, seed, pair_depth,
                            max_seconds, target, stop_on_target, external_stop);
    std::vector<Result> partial(static_cast<std::size_t>(workers));
    std::atomic<bool> local_stop{false};
    std::atomic<bool>* shared_stop = external_stop != nullptr ? external_stop : &local_stop;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    const int base = total_trials / workers;
    const int remainder = total_trials % workers;
    for (int t = 0; t < workers; ++t) {
        const int local_trials = base + (t < remainder ? 1 : 0);
        const std::uint64_t local_seed = seed +
            0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(t + 1);
        pool.emplace_back([&, t, local_trials, local_seed]() {
            partial[static_cast<std::size_t>(t)] = run_ris_fast(
                kernel, dual, n, local_trials, local_seed, pair_depth,
                max_seconds, target, stop_on_target, shared_stop);
        });
    }
    for (auto& worker : pool) worker.join();
    Result merged; merged.best = n + 1; merged.witness = Bits(n);
    for (const auto& result : partial) {
        merged.trials += result.trials;
        merged.timed_out = merged.timed_out || result.timed_out;
        merged.found_target = merged.found_target || result.found_target;
        if (result.best > 0 && result.best < merged.best) {
            merged.best = result.best; merged.witness = result.witness;
        }
    }
    return merged;
}

bool nontrivial(const Bits& row, const Rows& dual) {
    for (const auto& detector : dual) if (row.dot(detector)) return true;
    return false;
}

void consider(const Rows& rows, const Rows& dual, int pair_depth, Result& result) {
    const bool require_logical = !dual.empty();
    std::vector<int> weights;
    weights.reserve(rows.size());
    for (const auto& row : rows) {
        int weight = row.weight();
        weights.push_back(weight);
        if (weight > 0 && weight < result.best &&
            (!require_logical || nontrivial(row, dual))) {
            result.best = weight;
            result.witness = row;
        }
    }
    if (pair_depth <= 1 || rows.size() < 2) return;
    std::vector<int> light(rows.size());
    std::iota(light.begin(), light.end(), 0);
    std::stable_sort(light.begin(), light.end(), [&](int a, int b) {
        return weights[static_cast<std::size_t>(a)] < weights[static_cast<std::size_t>(b)];
    });
    light.resize(std::min<std::size_t>(static_cast<std::size_t>(pair_depth), light.size()));
    for (std::size_t a = 0; a + 1 < light.size(); ++a) {
        for (std::size_t b = a + 1; b < light.size(); ++b) {
            Bits combined = rows[static_cast<std::size_t>(light[a])];
            combined ^= rows[static_cast<std::size_t>(light[b])];
            const int weight = combined.weight();
            if (weight > 0 && weight < result.best &&
                (!require_logical || nontrivial(combined, dual))) {
                result.best = weight;
                result.witness = std::move(combined);
            }
        }
    }
}

Result run_ris(Rows kernel, const Rows& dual, int n, int trials,
               std::uint64_t seed, int pair_depth, double max_seconds,
               int target, bool stop_on_target) {
    Result result;
    result.best = n + 1;
    result.witness = Bits(n);
    std::mt19937_64 rng(seed);
    std::vector<int> order = identity_order(n);
    const auto start = Clock::now();
    const bool has_deadline = max_seconds >= 0.0;
    const auto deadline = start + std::chrono::duration<double>(std::max(0.0, max_seconds));
    for (int trial = 0; trial < std::max(0, trials); ++trial) {
        std::shuffle(order.begin(), order.end(), rng);
        Rows reduced = rref(kernel, n, order);
        consider(reduced, dual, pair_depth, result);
        result.trials = trial + 1;
        if (target > 0 && result.best < target && stop_on_target) {
            result.found_target = true;
            break;
        }
        if (has_deadline && ((trial + 1) & 63) == 0 && Clock::now() > deadline) {
            result.timed_out = true;
            break;
        }
    }
    return result;
}

Result run_ris_parallel(const Rows& kernel, const Rows& dual, int n, int trials,
                        std::uint64_t seed, int pair_depth, double max_seconds,
                        int target, bool stop_on_target, int requested_threads) {
    const int total_trials = std::max(0, trials);
    if (total_trials == 0) {
        Result empty;
        empty.best = n + 1;
        empty.witness = Bits(n);
        return empty;
    }
    int hardware = static_cast<int>(std::thread::hardware_concurrency());
    if (hardware < 1) hardware = 1;
    const int workers = std::max(1, std::min(total_trials,
        requested_threads > 0 ? requested_threads : hardware));
    if (workers == 1) {
        return run_ris(kernel, dual, n, total_trials, seed, pair_depth,
                       max_seconds, target, stop_on_target);
    }

    std::vector<Result> partial(static_cast<std::size_t>(workers));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    const int base = total_trials / workers;
    const int remainder = total_trials % workers;
    for (int t = 0; t < workers; ++t) {
        const int local_trials = base + (t < remainder ? 1 : 0);
        const std::uint64_t local_seed = seed +
            0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(t + 1);
        pool.emplace_back([&, t, local_trials, local_seed]() {
            partial[static_cast<std::size_t>(t)] = run_ris(
                kernel, dual, n, local_trials, local_seed, pair_depth,
                max_seconds, target, stop_on_target);
        });
    }
    for (auto& worker : pool) worker.join();

    Result merged;
    merged.best = n + 1;
    merged.witness = Bits(n);
    for (const auto& result : partial) {
        merged.trials += result.trials;
        merged.timed_out = merged.timed_out || result.timed_out;
        merged.found_target = merged.found_target || result.found_target;
        if (result.best > 0 && result.best < merged.best) {
            merged.best = result.best;
            merged.witness = result.witness;
        }
    }
    return merged;
}

void write_side(const Result& result, int n, int* best, int* trials,
                int* timed_out, unsigned char* witness) {
    *best = (result.best == n + 1) ? 0 : result.best;
    *trials = result.trials;
    *timed_out = result.timed_out ? 1 : 0;
    std::fill(witness, witness + n, static_cast<unsigned char>(0));
    for (int c = 0; c < n; ++c) if (result.witness.test(c)) witness[c] = 1;
}

void append_rows(Rows& dst, Rows src) {
    for (auto& row : src) dst.push_back(std::move(row));
}

Bits and_not_bits(const Bits& a, const Bits& b) {
    Bits out = a;
    for (std::size_t i = 0; i < out.w.size(); ++i) out.w[i] &= ~b.w[i];
    return out;
}

bool intersects(const Bits& a, const Bits& b) {
    for (std::size_t i = 0; i < a.w.size(); ++i) if (a.w[i] & b.w[i]) return true;
    return false;
}

using MaskKey = std::array<std::uint64_t, 6>;

struct ExactBits {
    MaskKey w{};
    bool test(int bit) const { return ((w[static_cast<std::size_t>(bit >> 6)] >> (bit & 63)) & 1U) != 0; }
    void set(int bit) { w[static_cast<std::size_t>(bit >> 6)] |= 1ULL << (bit & 63); }
    bool any() const { for (auto x : w) if (x) return true; return false; }
    int weight() const { int out = 0; for (auto x : w) out += popcount64(x); return out; }
    ExactBits& operator^=(const ExactBits& other) {
        for (std::size_t i = 0; i < w.size(); ++i) w[i] ^= other.w[i];
        return *this;
    }
};

MaskKey mask_key(const Bits& bits) {
    MaskKey key{};
    for (std::size_t i = 0; i < std::min<std::size_t>(key.size(), bits.w.size()); ++i) key[i] = bits.w[i];
    return key;
}

MaskKey mask_key(const ExactBits& bits) { return bits.w; }

ExactBits and_not_bits(const ExactBits& a, const ExactBits& b) {
    ExactBits out = a;
    for (std::size_t i = 0; i < out.w.size(); ++i) out.w[i] &= ~b.w[i];
    return out;
}

bool intersects(const ExactBits& a, const ExactBits& b) {
    for (std::size_t i = 0; i < a.w.size(); ++i) if (a.w[i] & b.w[i]) return true;
    return false;
}

struct MaskHash {
    std::size_t operator()(const MaskKey& key) const noexcept {
        std::size_t h = 1469598103934665603ULL;
        for (auto x : key) {
            h ^= static_cast<std::size_t>(x);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

class MaskSet {
public:
    MaskSet() : slots(1U << 16), used(slots.size(), 0) {}

    bool contains(const MaskKey& key) const {
        if (slots.empty()) return false;
        std::size_t i = MaskHash{}(key) & (slots.size() - 1);
        for (;;) {
            if (!used[i]) return false;
            if (slots[i] == key) return true;
            i = (i + 1) & (slots.size() - 1);
        }
    }

    bool insert(const MaskKey& key) {
        if ((count + 1) * 10 >= slots.size() * 7) rehash(slots.size() * 2);
        return insert_raw(key);
    }

private:
    std::vector<MaskKey> slots;
    std::vector<unsigned char> used;
    std::size_t count = 0;

    bool insert_raw(const MaskKey& key) {
        std::size_t i = MaskHash{}(key) & (slots.size() - 1);
        for (;;) {
            if (!used[i]) {
                slots[i] = key;
                used[i] = 1;
                ++count;
                return true;
            }
            if (slots[i] == key) return false;
            i = (i + 1) & (slots.size() - 1);
        }
    }

    void rehash(std::size_t capacity) {
        std::vector<MaskKey> old_slots = std::move(slots);
        std::vector<unsigned char> old_used = std::move(used);
        slots.assign(capacity, MaskKey{});
        used.assign(capacity, 0);
        count = 0;
        for (std::size_t i = 0; i < old_slots.size(); ++i)
            if (old_used[i]) insert_raw(old_slots[i]);
    }
};

struct ExactResult {
    int status = 0; // 0 certified_none, 1 found, 2 timeout
    std::int64_t nodes = 0;
    int starts_done = 0;
    ExactBits witness;
    ExactBits signature;
};

class ExactTannerSearch {
public:
    ExactTannerSearch(const unsigned char* matrix, int matrix_rows,
                      const unsigned char* dual, int dual_rows, int cols,
                      int threshold, double seconds, bool degree_desc)
        : n(cols), m(std::max(0, matrix_rows)), drows(std::max(0, dual_rows)),
          cap(threshold), deadline_seconds(seconds), witness(), signature() {
        perm.resize(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), 0);
        if (degree_desc) {
            std::vector<int> degrees(static_cast<std::size_t>(n), 0);
            for (int v = 0; v < n; ++v)
                for (int r = 0; r < m; ++r)
                    degrees[static_cast<std::size_t>(v)] +=
                        (matrix[static_cast<std::size_t>(r) * n + v] & 1U) != 0;
            std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
                return degrees[static_cast<std::size_t>(a)] > degrees[static_cast<std::size_t>(b)];
            });
        }
        check_masks.reserve(static_cast<std::size_t>(m));
        for (int r = 0; r < m; ++r) {
            ExactBits row;
            for (int internal = 0; internal < n; ++internal) {
                const int physical = perm[static_cast<std::size_t>(internal)];
                if (matrix[static_cast<std::size_t>(r) * n + physical] & 1U) row.set(internal);
            }
            check_masks.push_back(std::move(row));
        }
        col_syn.reserve(static_cast<std::size_t>(n));
        col_sig.reserve(static_cast<std::size_t>(n));
        col_deg.reserve(static_cast<std::size_t>(n));
        for (int internal = 0; internal < n; ++internal) {
            const int physical = perm[static_cast<std::size_t>(internal)];
            ExactBits syn, sig;
            for (int r = 0; r < m; ++r)
                if (matrix[static_cast<std::size_t>(r) * n + physical] & 1U) syn.set(r);
            for (int r = 0; r < drows; ++r)
                if (dual[static_cast<std::size_t>(r) * n + physical] & 1U) sig.set(r);
            col_deg.push_back(syn.weight());
            col_syn.push_back(std::move(syn));
            col_sig.push_back(std::move(sig));
        }
        maxdeg = 1;
        for (int degree : col_deg) maxdeg = std::max(maxdeg, degree);
        all_vars = ExactBits();
        for (int v = 0; v < n; ++v) all_vars.set(v);
    }

    ExactResult run() {
        ExactResult out;
        out.witness = ExactBits();
        out.signature = ExactBits();
        if (cap < 1) { out.starts_done = n; return out; }
        start_time = Clock::now();
        for (int start = 0; start < n; ++start) {
            ExactBits avail = all_vars;
            for (int v = 0; v <= start; ++v) avail.w[static_cast<std::size_t>(v >> 6)] &= ~(1ULL << (v & 63));
            ExactBits selected; selected.set(start);
            memo.insert(mask_key(selected));
            ExactBits found, found_sig;
            if (dfs(selected, col_syn[static_cast<std::size_t>(start)],
                    col_sig[static_cast<std::size_t>(start)], 1, avail,
                    found, found_sig)) {
                out.status = 1;
                out.nodes = nodes;
                out.starts_done = starts_done;
                out.witness = std::move(found);
                out.signature = std::move(found_sig);
                return out;
            }
            ++starts_done;
            if (timed_out) { out.status = 2; break; }
        }
        out.status = timed_out ? 2 : 0;
        out.nodes = nodes;
        out.starts_done = starts_done;
        return out;
    }

    int physical_index(int internal) const { return perm[static_cast<std::size_t>(internal)]; }

private:
    bool timed_out = false;
    std::int64_t nodes = 0;
    int starts_done = 0;
    const int n, m, drows, cap;
    const double deadline_seconds;
    int maxdeg = 1;
    std::vector<int> perm, col_deg;
    std::vector<ExactBits> check_masks, col_syn, col_sig;
    ExactBits all_vars, witness, signature;
    MaskSet memo;
    Clock::time_point start_time{};

    bool deadline_hit() {
        if (deadline_seconds < 0.0 || (nodes & 4095) != 0) return false;
        const double elapsed = std::chrono::duration<double>(Clock::now() - start_time).count();
        return elapsed > deadline_seconds;
    }

    bool analyze(const ExactBits& syn, const ExactBits& remaining,
                 ExactBits& best_mask, int& lower_bound) {
        bool have_best = false;
        int best_count = n + 1;
        ExactBits used;
        int packing = 0;
        for (int r = 0; r < m; ++r) {
            if (!syn.test(r)) continue;
            ExactBits cm = and_not_bits(check_masks[static_cast<std::size_t>(r)],
                                        and_not_bits(all_vars, remaining));
            const int count = cm.weight();
            if (count == 0) { lower_bound = cap + 1; return false; }
            if (count < best_count) { best_count = count; best_mask = cm; have_best = true; }
            if (!intersects(cm, used)) { ++packing; used ^= cm; }
        }
        const int degree_lb = (syn.weight() + maxdeg - 1) / maxdeg;
        lower_bound = std::max(degree_lb, packing);
        return have_best;
    }

    bool dfs(const ExactBits& selected, const ExactBits& syn, const ExactBits& sig,
             int weight, const ExactBits& avail, ExactBits& found, ExactBits& found_sig) {
        ++nodes;
        if (deadline_hit()) { timed_out = true; return false; }
        if (!syn.any()) {
            if (sig.any()) { found = selected; found_sig = sig; return true; }
            return false;
        }
        if (weight >= cap) return false;
        ExactBits remaining = and_not_bits(avail, selected);
        ExactBits branch_mask;
        int lower_bound = 0;
        if (!analyze(syn, remaining, branch_mask, lower_bound) || weight + lower_bound > cap) return false;
        for (int v = 0; v < n; ++v) {
            if (!branch_mask.test(v)) continue;
            ExactBits new_selected = selected; new_selected.set(v);
            if (memo.contains(mask_key(new_selected))) continue;
            memo.insert(mask_key(new_selected));
            ExactBits new_syn = syn; new_syn ^= col_syn[static_cast<std::size_t>(v)];
            ExactBits new_sig = sig; new_sig ^= col_sig[static_cast<std::size_t>(v)];
            if (dfs(new_selected, new_syn, new_sig, weight + 1, avail, found, found_sig)) return true;
            if (timed_out) return false;
        }
        return false;
    }
};

} // namespace

QLDPC_EXPORT int qldpc_ris_generator(
    const unsigned char* generator, int generator_rows, int n,
    const unsigned char* dual, int dual_rows, int trials, std::uint64_t seed,
    int pair_depth, double max_seconds, int target, int stop_on_target,
    int* best, int* trials_run, int* timed_out, unsigned char* witness) {
    try {
        Rows kernel = dense_rows(generator, generator_rows, n);
        Rows detectors = dense_rows(dual, dual_rows, n);
        Result result = run_ris(std::move(kernel), detectors, n, trials, seed,
                                pair_depth, max_seconds, target, stop_on_target != 0);
        write_side(result, n, best, trials_run, timed_out, witness);
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_gf2_rank(
    const unsigned char* matrix, int rows, int cols) {
    if (!matrix || rows < 0 || cols < 0) return -1;
    Rows reduced = rref(dense_rows(matrix, rows, cols), cols,
                        identity_order(cols));
    return static_cast<int>(reduced.size());
}

QLDPC_EXPORT int qldpc_batch_systematic(
    const int* mult, const int* inv, int s, const int* supports,
    int draws, int support_len, int left, unsigned char* generators,
    int* probes, int* valid) {
    try {
        const int generator_width = 2 * s;
        const std::size_t generator_stride = static_cast<std::size_t>(s) * generator_width;
        std::fill(generators, generators + static_cast<std::size_t>(std::max(0, draws)) * generator_stride,
                  static_cast<unsigned char>(0));
        for (int draw = 0; draw < std::max(0, draws); ++draw) {
            const int* pair = supports + static_cast<std::size_t>(draw) * 2 * support_len;
            Rows a = representation(mult, inv, s, pair, support_len, left != 0);
            Rows b = representation(mult, inv, s, pair + support_len, support_len, left != 0);
            Rows bi;
            if (!invert_square(b, s, bi)) {
                probes[draw] = 0;
                valid[draw] = 0;
                continue;
            }
            unsigned char* out = generators + static_cast<std::size_t>(draw) * generator_stride;
            for (int i = 0; i < s; ++i) out[static_cast<std::size_t>(i) * generator_width + i] = 1;
            Rows product_rows;
            product_rows.reserve(static_cast<std::size_t>(s));
            for (int j = 0; j < s; ++j) {
                Bits row(s);
                for (int r = 0; r < s; ++r) {
                    if (bi[static_cast<std::size_t>(j)].test(r)) row ^= a[static_cast<std::size_t>(r)];
                }
                product_rows.push_back(std::move(row));
            }
            for (int i = 0; i < s; ++i) {
                for (int j = 0; j < s; ++j) {
                    if (product_rows[static_cast<std::size_t>(j)].test(i)) {
                        out[static_cast<std::size_t>(i) * generator_width + s + j] = 1;
                    }
                }
            }
            Rows generator_rows;
            generator_rows.reserve(static_cast<std::size_t>(s));
            for (int i = 0; i < s; ++i) {
                Bits row(generator_width);
                for (int c = 0; c < generator_width; ++c) {
                    if (out[static_cast<std::size_t>(i) * generator_width + c]) row.set(c);
                }
                generator_rows.push_back(std::move(row));
            }
            probes[draw] = systematic_probe(generator_rows, generator_width);
            valid[draw] = 1;
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_cyclic_ideal_mine(
    const int* bases, int base_count, int support_len, int m,
    int iterations, int min_weight, int max_weight, std::uint64_t seed,
    int max_words, int* out_supports, int* out_weights, int* word_count) {
    try {
        if (m < 2 || m > 768 || base_count <= 0 || support_len <= 0 ||
            max_words <= 0 || max_weight <= 0 || max_weight > 768) {
            return 2;
        }
        const int capacity = std::min(max_words, 1000000);
        const int width = std::min(max_weight, m);
        std::vector<CyclicKey> shifted;
        shifted.reserve(static_cast<std::size_t>(base_count) * m);
        for (int base = 0; base < base_count; ++base) {
            const int* support = bases + static_cast<std::size_t>(base) * support_len;
            for (int shift = 0; shift < m; ++shift)
                shifted.push_back(shifted_cyclic_word(support, support_len, shift, m));
        }

        std::unordered_set<CyclicKey, CyclicKeyHash> seen;
        seen.reserve(static_cast<std::size_t>(capacity) * 2);
        int count = 0;
        auto emit = [&](const CyclicKey& key) {
            if (count >= capacity) return;
            const int weight = cyclic_popcount(key, m);
            if (weight < min_weight || weight > width) return;
            if (!seen.insert(key).second) return;
            int* dst = out_supports + static_cast<std::size_t>(count) * width;
            int at = 0;
            for (int p = 0; p < m && at < width; ++p)
                if ((key[static_cast<std::size_t>(p >> 6)] >> (p & 63)) & 1ULL)
                    dst[at++] = p;
            out_weights[count] = weight;
            ++count;
        };

        // Shift orbits and pair shells provide deterministic coverage of the
        // known ideal before stochastic higher-order mutations begin.
        for (const auto& key : shifted) emit(key);
        for (std::size_t i = 0; i < shifted.size() && count < capacity; ++i)
            for (std::size_t j = i + 1; j < shifted.size() && count < capacity; ++j)
                emit(xor_key(shifted[i], shifted[j]));

        std::mt19937_64 rng(seed);
        const int draws = std::max(0, iterations);
        for (int trial = 0; trial < draws && count < capacity; ++trial) {
            CyclicKey word{};
            const int terms = 2 + static_cast<int>(rng() % 15ULL);
            for (int term = 0; term < terms; ++term) {
                const std::size_t index = static_cast<std::size_t>(rng() % shifted.size());
                word = xor_key(word, shifted[index]);
            }
            emit(word);
        }
        *word_count = count;
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_cyclic_ideal_hillclimb(
    const int* base, int support_len, int m, int restarts, int steps,
    int min_weight, int max_weight, std::uint64_t seed, int max_words,
    int* out_supports, int* out_weights, int* word_count) {
    try {
        if (m < 2 || m > 768 || support_len <= 0 || restarts < 0 ||
            steps < 0 || max_words <= 0 || min_weight < 0 || max_weight <= 0 ||
            max_weight > 768 || min_weight > max_weight) return 2;
        const int capacity = std::min(max_words, 1000000);
        const int width = std::min(max_weight, m);
        std::vector<CyclicKey> shifted;
        shifted.reserve(static_cast<std::size_t>(m));
        for (int shift = 0; shift < m; ++shift)
            shifted.push_back(shifted_cyclic_word(base, support_len, shift, m));
        std::unordered_set<CyclicKey, CyclicKeyHash> seen;
        seen.reserve(static_cast<std::size_t>(capacity) * 2);
        int count = 0;
        auto emit = [&](const CyclicKey& key) {
            if (count >= capacity) return;
            const int weight = cyclic_popcount(key, m);
            if (weight < min_weight || weight > width || weight == 0) return;
            if (!seen.insert(key).second) return;
            int* dst = out_supports + static_cast<std::size_t>(count) * width;
            int at = 0;
            for (int p = 0; p < m && at < width; ++p)
                if ((key[static_cast<std::size_t>(p >> 6)] >> (p & 63)) & 1ULL)
                    dst[at++] = p;
            out_weights[count] = weight;
            ++count;
        };

        std::mt19937_64 rng(seed);
        const int probes = std::min(32, std::max(4, m));
        for (int restart = 0; restart < restarts && count < capacity; ++restart) {
            CyclicKey current{};
            const int terms = 1 + static_cast<int>(rng() %
                static_cast<std::uint64_t>(std::min(m, 128)));
            for (int term = 0; term < terms; ++term)
                current = xor_key(current,
                    shifted[static_cast<std::size_t>(rng() % static_cast<std::uint64_t>(m))]);
            if (cyclic_popcount(current, m) == 0) continue;
            int current_weight = cyclic_popcount(current, m);
            CyclicKey local_best = current;
            int local_best_weight = current_weight;
            emit(current);
            for (int step = 0; step < steps && count < capacity; ++step) {
                CyclicKey proposal{};
                int proposal_weight = m + 1;
                // Best-of-random coordinate move: much stronger than a plain
                // random XOR when a sparse word needs many lift terms.
                for (int probe = 0; probe < probes; ++probe) {
                    const auto& candidate = shifted[static_cast<std::size_t>(
                        rng() % static_cast<std::uint64_t>(m))];
                    CyclicKey trial = xor_key(current, candidate);
                    const int weight = cyclic_popcount(trial, m);
                    if (weight < proposal_weight) {
                        proposal = trial;
                        proposal_weight = weight;
                    }
                }
                bool accept = proposal_weight <= current_weight;
                if (!accept) {
                    // Small uphill moves escape coordinate minima; the
                    // probability cools as the restart progresses.
                    const int allowance = std::max(50, 2500 -
                        (step * 2400) / std::max(1, steps));
                    accept = static_cast<int>(rng() % 10000ULL) < allowance;
                }
                if (accept && proposal_weight > 0) {
                    current = proposal;
                    current_weight = proposal_weight;
                }
                if (current_weight < local_best_weight) {
                    local_best = current;
                    local_best_weight = current_weight;
                }
                emit(current);
                if ((step & 255) == 255 && local_best_weight < current_weight) {
                    current = local_best;
                    current_weight = local_best_weight;
                }
            }
            emit(local_best);
        }
        *word_count = count;
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_css_ris(
    const unsigned char* hx, int hx_rows, const unsigned char* hz, int hz_rows, int n,
    int trials, std::uint64_t seed, int pair_depth, double max_seconds_per_side,
    int target, int stop_on_target, int* dx, int* dz, int* x_trials, int* z_trials,
    int* x_timed_out, int* z_timed_out, unsigned char* x_witness,
    unsigned char* z_witness, int* refuted) {
    try {
        // X logicals live in ker(H_Z) and are detected by Z-logical reps.
        Rows lx = logical_basis(hz, hz_rows, hx, hx_rows, n);
        Rows lz = logical_basis(hx, hx_rows, hz, hz_rows, n);
        Result x = run_ris(kernel_basis(hz, hz_rows, n), lz, n, trials, seed,
                           pair_depth, max_seconds_per_side, target, stop_on_target != 0);
        Result z = run_ris(kernel_basis(hx, hx_rows, n), lx, n, trials, seed + 1,
                           pair_depth, max_seconds_per_side, target, stop_on_target != 0);
        write_side(x, n, dx, x_trials, x_timed_out, x_witness);
        write_side(z, n, dz, z_trials, z_timed_out, z_witness);
        *refuted = (target > 0 &&
                    ((x.best > 0 && x.best < target) ||
                     (z.best > 0 && z.best < target))) ? 1 : 0;
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_css_ris_parallel(
    const unsigned char* hx, int hx_rows, const unsigned char* hz, int hz_rows, int n,
    int trials, std::uint64_t seed, int pair_depth, double max_seconds_per_side,
    int target, int stop_on_target, int requested_threads, int* dx, int* dz,
    int* x_trials, int* z_trials, int* x_timed_out, int* z_timed_out,
    unsigned char* x_witness, unsigned char* z_witness, int* refuted) {
    try {
        // Precompute both complete kernels once. Run sectors concurrently, then
        // split requested worker count across them to avoid oversubscription.
        Rows lx = logical_basis(hz, hz_rows, hx, hx_rows, n);
        Rows lz = logical_basis(hx, hx_rows, hz, hz_rows, n);
        Rows x_kernel = kernel_basis(hz, hz_rows, n);
        Rows z_kernel = kernel_basis(hx, hx_rows, n);
        const int hardware = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        const int total_threads = requested_threads > 0 ? requested_threads : hardware;
        const int x_threads = std::max(1, (total_threads + 1) / 2);
        const int z_threads = std::max(1, total_threads / 2);
        FastRows x_fast = to_fast_rows(x_kernel);
        FastRows z_fast = to_fast_rows(z_kernel);
        FastRows x_dual = to_fast_rows(lz);
        FastRows z_dual = to_fast_rows(lx);
        Result x, z;
        std::atomic<bool> global_stop{false};
        std::thread x_worker([&]() {
            x = run_ris_fast_parallel(x_fast, x_dual, n, trials, seed, pair_depth,
                                      max_seconds_per_side, target,
                                      stop_on_target != 0, x_threads, &global_stop);
        });
        std::thread z_worker([&]() {
            z = run_ris_fast_parallel(z_fast, z_dual, n, trials, seed + 1, pair_depth,
                                      max_seconds_per_side, target,
                                      stop_on_target != 0, z_threads, &global_stop);
        });
        x_worker.join();
        z_worker.join();
        write_side(x, n, dx, x_trials, x_timed_out, x_witness);
        write_side(z, n, dz, z_trials, z_timed_out, z_witness);
        *refuted = (target > 0 &&
                    ((x.best > 0 && x.best < target) ||
                     (z.best > 0 && z.best < target))) ? 1 : 0;
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_css_ris_spanned(
    const unsigned char* hx, int hx_rows, const unsigned char* hz, int hz_rows,
    const unsigned char* lx, int lx_rows, const unsigned char* lz, int lz_rows, int n,
    int trials, std::uint64_t seed, int pair_depth, double max_seconds_per_side,
    int target, int stop_on_target, int* dx, int* dz, int* x_trials, int* z_trials,
    int* x_timed_out, int* z_timed_out, unsigned char* x_witness,
    unsigned char* z_witness, int* refuted) {
    try {
        // Caller supplies quotient bases.  Stabilizers + logicals span each
        // complete kernel, avoiding repeated nullspace/logical elimination in
        // product-search hot loops.
        Rows x_kernel = dense_rows(hx, hx_rows, n);
        Rows z_kernel = dense_rows(hz, hz_rows, n);
        append_rows(x_kernel, dense_rows(lx, lx_rows, n));
        append_rows(z_kernel, dense_rows(lz, lz_rows, n));
        Rows x_dual = dense_rows(lz, lz_rows, n);
        Rows z_dual = dense_rows(lx, lx_rows, n);
        Result x = run_ris(std::move(x_kernel), x_dual, n, trials, seed,
                           pair_depth, max_seconds_per_side, target, stop_on_target != 0);
        Result z = run_ris(std::move(z_kernel), z_dual, n, trials, seed + 1,
                           pair_depth, max_seconds_per_side, target, stop_on_target != 0);
        write_side(x, n, dx, x_trials, x_timed_out, x_witness);
        write_side(z, n, dz, z_trials, z_timed_out, z_witness);
        *refuted = (target > 0 &&
                    ((x.best > 0 && x.best < target) ||
                     (z.best > 0 && z.best < target))) ? 1 : 0;
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_partition_anneal(
    const unsigned char* checks, int check_rows, int n,
    const int* initial_labels, const unsigned char* frozen,
    const unsigned char* must_active, const unsigned char* tags,
    int block_count, int steps, std::uint64_t seed,
    int target_odd, int* labels_out, int* max_odd_out, int* cost_out) {
    try {
        if (!checks || !initial_labels || !labels_out || !max_odd_out || !cost_out ||
            check_rows < 0 || n <= 0 || block_count <= 0 || steps < 0 || target_odd < 0)
            return 2;
        std::vector<int> labels(static_cast<std::size_t>(n));
        for (int c = 0; c < n; ++c) {
            if (initial_labels[c] < 0 || initial_labels[c] >= block_count) return 2;
            if (must_active && (must_active[c] & 1U) && initial_labels[c] == 0) return 2;
            labels[static_cast<std::size_t>(c)] = initial_labels[c];
        }
        std::vector<int> label_x(static_cast<std::size_t>(block_count), 0);
        std::vector<int> label_z(static_cast<std::size_t>(block_count), 0);
        if (tags) {
            for (int c = 0; c < n; ++c) {
                const int label = labels[static_cast<std::size_t>(c)];
                if (label == 0) continue;
                label_x[static_cast<std::size_t>(label)] += (tags[c] & 1U) ? 1 : 0;
                label_z[static_cast<std::size_t>(label)] += (tags[c] & 2U) ? 1 : 0;
            }
            for (int label = 1; label < block_count; ++label) {
                if (label_x[static_cast<std::size_t>(label)] > 1 ||
                    label_z[static_cast<std::size_t>(label)] > 1) return 2;
            }
        }
        std::vector<std::vector<int>> supports(static_cast<std::size_t>(check_rows));
        std::vector<std::vector<int>> incidence(static_cast<std::size_t>(n));
        for (int r = 0; r < check_rows; ++r) {
            for (int c = 0; c < n; ++c) {
                if (checks[static_cast<std::size_t>(r) * n + c] & 1U) {
                    supports[static_cast<std::size_t>(r)].push_back(c);
                    incidence[static_cast<std::size_t>(c)].push_back(r);
                }
            }
        }
        std::vector<unsigned char> parity(
            static_cast<std::size_t>(check_rows) * block_count, 0);
        for (int r = 0; r < check_rows; ++r) {
            for (int c : supports[static_cast<std::size_t>(r)]) {
                const int label = labels[static_cast<std::size_t>(c)];
                parity[static_cast<std::size_t>(r) * block_count + label] ^= 1U;
            }
        }
        std::vector<int> odd(static_cast<std::size_t>(check_rows), 0);
        auto penalty = [target_odd](int value) {
            const int excess = std::max(0, value - target_odd);
            return excess * excess;
        };
        int cost = 0;
        int max_odd = 0;
        for (int r = 0; r < check_rows; ++r) {
            int value = 0;
            for (int b = 0; b < block_count; ++b)
                value += parity[static_cast<std::size_t>(r) * block_count + b] ? 1 : 0;
            odd[static_cast<std::size_t>(r)] = value;
            cost += penalty(value);
            max_odd = std::max(max_odd, value);
        }
        std::vector<int> best_labels = labels;
        int best_cost = cost;
        int best_max_odd = max_odd;
        std::mt19937_64 rng(seed);
        auto movable = [frozen](int coordinate) {
            return !frozen || !(frozen[coordinate] & 1U);
        };
        std::vector<int> affected;
        affected.reserve(128);
        for (int step = 0; step < steps; ++step) {
            int a = -1;
            if ((step & 63) == 0 && check_rows > 0) {
                int worst = 0;
                for (int r = 1; r < check_rows; ++r) {
                    if (odd[static_cast<std::size_t>(r)] > odd[static_cast<std::size_t>(worst)])
                        worst = r;
                }
                const auto& support = supports[static_cast<std::size_t>(worst)];
                for (std::size_t tries = 0; tries < support.size(); ++tries) {
                    const int candidate = support[static_cast<std::size_t>(rng() % support.size())];
                    if (movable(candidate)) {
                        a = candidate;
                        break;
                    }
                }
            } else {
                for (int tries = 0; tries < 32; ++tries) {
                    const int candidate = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
                    if (movable(candidate)) {
                        a = candidate;
                        break;
                    }
                }
            }
            if (a < 0) continue;
            const int label_a = labels[static_cast<std::size_t>(a)];
            int b = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
            for (int tries = 0; (!movable(b) || labels[static_cast<std::size_t>(b)] == label_a) && tries < 64; ++tries)
                b = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
            if (!movable(b) || labels[static_cast<std::size_t>(b)] == label_a) continue;
            const int label_b = labels[static_cast<std::size_t>(b)];
            if ((must_active && (must_active[a] & 1U) && label_b == 0) ||
                (must_active && (must_active[b] & 1U) && label_a == 0)) continue;
            if (tags) {
                const int ax = (tags[a] & 1U) ? 1 : 0;
                const int az = (tags[a] & 2U) ? 1 : 0;
                const int bx = (tags[b] & 1U) ? 1 : 0;
                const int bz = (tags[b] & 2U) ? 1 : 0;
                if ((label_a != 0 &&
                     (label_x[static_cast<std::size_t>(label_a)] - ax + bx > 1 ||
                      label_z[static_cast<std::size_t>(label_a)] - az + bz > 1)) ||
                    (label_b != 0 &&
                     (label_x[static_cast<std::size_t>(label_b)] - bx + ax > 1 ||
                      label_z[static_cast<std::size_t>(label_b)] - bz + az > 1))) continue;
            }
            affected.clear();
            const auto& left = incidence[static_cast<std::size_t>(a)];
            const auto& right = incidence[static_cast<std::size_t>(b)];
            std::size_t i = 0, j = 0;
            while (i < left.size() || j < right.size()) {
                if (j == right.size() || (i < left.size() && left[i] < right[j])) {
                    affected.push_back(left[i++]);
                } else if (i == left.size() || right[j] < left[i]) {
                    affected.push_back(right[j++]);
                } else {
                    ++i;
                    ++j;
                }
            }
            int delta = 0;
            for (int r : affected) {
                const std::size_t base = static_cast<std::size_t>(r) * block_count;
                const int old_value = odd[static_cast<std::size_t>(r)];
                const bool pa = parity[base + label_a] != 0;
                const bool pb = parity[base + label_b] != 0;
                const int new_value = old_value + ((!pa && !pb) ? 2 : (pa && pb) ? -2 : 0);
                delta += penalty(new_value) - penalty(old_value);
            }
            bool accept = delta <= 0;
            if (!accept) {
                const int cooling = std::max(20, 2200 -
                    (step * 2150) / std::max(1, steps));
                const int scale = std::max(1, delta);
                accept = static_cast<int>(rng() % static_cast<std::uint64_t>(10000 * scale)) < cooling;
            }
            if (!accept) continue;
            labels[static_cast<std::size_t>(a)] = label_b;
            labels[static_cast<std::size_t>(b)] = label_a;
            if (tags) {
                const int ax = (tags[a] & 1U) ? 1 : 0;
                const int az = (tags[a] & 2U) ? 1 : 0;
                const int bx = (tags[b] & 1U) ? 1 : 0;
                const int bz = (tags[b] & 2U) ? 1 : 0;
                if (label_a != 0) {
                    label_x[static_cast<std::size_t>(label_a)] += bx - ax;
                    label_z[static_cast<std::size_t>(label_a)] += bz - az;
                }
                if (label_b != 0) {
                    label_x[static_cast<std::size_t>(label_b)] += ax - bx;
                    label_z[static_cast<std::size_t>(label_b)] += az - bz;
                }
            }
            for (int r : affected) {
                const std::size_t base = static_cast<std::size_t>(r) * block_count;
                const bool pa = parity[base + label_a] != 0;
                const bool pb = parity[base + label_b] != 0;
                if (!pa && !pb) odd[static_cast<std::size_t>(r)] += 2;
                else if (pa && pb) odd[static_cast<std::size_t>(r)] -= 2;
                parity[base + label_a] ^= 1U;
                parity[base + label_b] ^= 1U;
            }
            cost += delta;
            if ((step & 1023) == 0 || cost < best_cost) {
                max_odd = 0;
                for (int value : odd) max_odd = std::max(max_odd, value);
                if (cost < best_cost || (cost == best_cost && max_odd < best_max_odd)) {
                    best_cost = cost;
                    best_max_odd = max_odd;
                    best_labels = labels;
                }
            }
        }
        std::copy(best_labels.begin(), best_labels.end(), labels_out);
        *max_odd_out = best_max_odd;
        *cost_out = best_cost;
        return 0;
    } catch (...) {
        return 1;
    }
}

QLDPC_EXPORT int qldpc_exact_logical(
    const unsigned char* matrix, int matrix_rows, const unsigned char* dual,
    int dual_rows, int n, int cap, double max_seconds, int degree_desc,
    int* status, std::int64_t* nodes, int* starts_done,
    unsigned char* witness, int* witness_weight, unsigned char* signature,
    int* signature_len) {
    try {
        ExactTannerSearch search(matrix, matrix_rows, dual, dual_rows, n,
                                  cap, max_seconds, degree_desc != 0);
        ExactResult result = search.run();
        *status = result.status;
        *nodes = result.nodes;
        *starts_done = result.starts_done;
        *witness_weight = result.status == 1 ? result.witness.weight() : 0;
        *signature_len = result.status == 1 ? result.signature.weight() : 0;
        std::fill(witness, witness + n, static_cast<unsigned char>(0));
        std::fill(signature, signature + dual_rows, static_cast<unsigned char>(0));
        if (result.status == 1) {
            for (int internal = 0; internal < n; ++internal)
                if (result.witness.test(internal)) witness[search.physical_index(internal)] = 1;
            for (int r = 0; r < dual_rows; ++r)
                if (result.signature.test(r)) signature[r] = 1;
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
