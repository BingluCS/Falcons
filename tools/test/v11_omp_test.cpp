// Standalone correctness checks; compile on the test server with C++17/OpenMP.
// Add -mavx2 -mfma for AVX2, or omit them to exercise the scalar fallback.
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "Falcons/decomposition/Decomposition.hpp"
#include "Falcons/quantizer/Quantizer_Omp.hpp"
#include "Falcons/utils/BlockwiseIterator.hpp"
#include "Falcons/utils/FileUtil.hpp"
#include "Falcons/utils/InterpolationPacking.hpp"
#include "Falcons/utils/Interpolators.hpp"
#include "Falcons/utils/Iterator.hpp"
#include "Falcons/utils/MemoryUtil.hpp"
#include "Falcons/utils/Timer.hpp"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

#ifndef _OPENMP
int main() { std::puts("SKIP: OpenMP required"); return 77; }
#else
#include <omp.h>
// Dependencies are already included; expose only the decomposition test state.
#define private public
#include "Falcons/quantizer/LinearQuantizer_Omp.hpp"
#include "Falcons/decomposition/InterpolationDecomposition_Omp.hpp"
#undef private

namespace {
template <class T, unsigned N = 3>
using Decomp = Falcons::InterpolationDecomposition_OMP<T, N, Falcons::LinearQuantizerOMP<T>>;
using Bytes = std::vector<unsigned char>;
static_assert(!std::is_copy_constructible_v<Falcons::LinearQuantizerOMP<float>>);
static_assert(!std::is_copy_assignable_v<Falcons::LinearQuantizerOMP<float>>);
static_assert(std::is_move_constructible_v<Falcons::LinearQuantizerOMP<float>>);
static_assert(std::is_move_assignable_v<Falcons::LinearQuantizerOMP<float>>);
static_assert(!std::is_copy_constructible_v<Decomp<float, 1>>);
static_assert(!std::is_copy_assignable_v<Decomp<float, 1>>);
static_assert(std::is_move_constructible_v<Decomp<float, 1>>);
static_assert(std::is_move_assignable_v<Decomp<float, 1>>);
size_t checks = 0;
size_t cases = 0;
std::string context;

void require(bool condition, const char *message) {
    ++checks;
    if (!condition) throw std::runtime_error(context + ": " + message);
}

void require_stream_capacity(size_t tid, int codes, int outliers,
                             size_t code_capacity, size_t outlier_capacity) {
    ++checks;
    if (static_cast<size_t>(codes) > code_capacity || static_cast<size_t>(outliers) > outlier_capacity) {
        char detail[192];
        std::snprintf(detail, sizeof(detail),
                      "stream capacity tid=%zu codes=%d/%zu outliers=%d/%zu",
                      tid, codes, code_capacity, outliers, outlier_capacity);
        throw std::runtime_error(context + ": " + detail);
    }
}

template <class T>
void require_reconstruction(size_t index, T original, T encoded, T recovered, double bound) {
    ++checks;
    const double error = std::abs(static_cast<double>(recovered) - static_cast<double>(original));
    if (!(error <= bound)) {
        char detail[320];
        std::snprintf(detail, sizeof(detail),
                      "same-build reconstruction bound index=%zu original=%.17g encoded=%.17g "
                      "recovered=%.17g error=%.17g bound=%.17g",
                      index, static_cast<double>(original), static_cast<double>(encoded),
                      static_cast<double>(recovered), error, bound);
        throw std::runtime_error(context + ": " + detail);
    }
}

void scalar_capacity_checks() {
    context = "scalar stream capacity";
    Falcons::Config conf(33, 35, 37);
    Decomp<int32_t> d(conf, Falcons::LinearQuantizerOMP<int32_t>(0.125));
    d.original_dimensions = {33, 35, 37};
    d.interp_id = 1;
    d.direction_sequence_id = 5;
    d.nThreads = 32;
    d.init();
    require(d.scalar_stream_capacity() == 6539, "32-thread cubic boundary-plane regression");
    d.nThreads = 1;
    require(d.scalar_stream_capacity() == conf.num, "one-thread bound counts every sample");
    d.anchor_stride = 8;
    d.init();
    require(d.scalar_stream_capacity() == conf.num, "anchors and codes share the capacity bound");
}

void simd_capacity_checks() {
    context = "SIMD stream capacity 2D";
    Falcons::Config conf2(34, 65);
    Decomp<double, 2> d2(conf2, Falcons::LinearQuantizerOMP<double>(0.001));
    d2.original_dimensions = {34, 65};
    d2.nThreads = 32;
    d2.interp_id = 1;
    d2.direction_sequence_id = 1;
    d2.anchor_stride = 8;
    d2.init();
    const size_t expected = omp_get_thread_limit() < 32 ? conf2.num : 475;
    require(d2.simd_stream_capacity() == expected, "32-thread cubic anchored regression");
    d2.nThreads = 1;
    for (int algo : {0, 1}) {
        for (int direction : {0, 1}) {
            for (size_t anchor : {size_t(0), size_t(8)}) {
                d2.interp_id = algo;
                d2.direction_sequence_id = direction;
                d2.anchor_stride = anchor;
                d2.init();
                require(d2.simd_stream_capacity() == conf2.num, "2D one-thread sample accounting");
            }
        }
    }
    d2.original_dimensions = {1, 257};
    d2.nThreads = 32;
    d2.interp_id = 1;
    d2.direction_sequence_id = 1;
    d2.anchor_stride = 0;
    d2.init();
    require(d2.simd_stream_capacity() > d2.scalar_stream_capacity(),
            "2D SIMD row ownership differs from scalar point ownership");

    context = "SIMD stream capacity 3D skewed";
    Falcons::Config conf3(2, 33, 65);
    Decomp<float> d3(conf3, Falcons::LinearQuantizerOMP<float>(0.001));
    d3.original_dimensions = {2, 33, 65};
    for (int algo : {0, 1}) {
        for (int direction : {0, 5}) {
            for (size_t anchor : {size_t(0), size_t(8)}) {
                d3.interp_id = algo;
                d3.direction_sequence_id = direction;
                d3.anchor_stride = anchor;
                d3.nThreads = 1;
                d3.init();
                require(d3.simd_stream_capacity() == conf3.num, "3D one-thread sample accounting");
                d3.nThreads = 32;
                require(d3.simd_stream_capacity() > (conf3.num / 32) * 4,
                        "3D skew exceeds four-times-average estimate");
            }
        }
    }
}

void partition_checks(int threads) {
    context = "logical partition P=" + std::to_string(threads);
    Falcons::Config conf(33, 35, 67);
    Decomp<float> d(conf, Falcons::LinearQuantizerOMP<float>(0.001));
    d.nThreads = threads;
    d.coarse_serial_ready = true;
    const size_t max = std::numeric_limits<size_t>::max();
    require(d.coarse_trip_count(7, 4, 2) == 0, "empty range");
    require(d.coarse_trip_count(1, max, 2) == max / 2, "large trip count");
    require(!d.coarse_region_is_small(max, max, max), "overflow workload rejected");
#ifdef Falcons_OMP_COARSE_TEST_MAX_VALUES
    constexpr size_t ceiling = Falcons_OMP_COARSE_TEST_MAX_VALUES;
#else
    constexpr size_t ceiling = 8192;
#endif
    const size_t limit = std::min(ceiling, static_cast<size_t>(threads) * 256);
    require(d.coarse_region_is_small(limit, 1) == (ceiling != 0), "inclusive workload threshold");
    require(!d.coarse_region_is_small(limit + 1, 1), "exclusive workload threshold");
    for (size_t count : {size_t(0), size_t(1), size_t(threads - 1),
                         size_t(threads), size_t(threads + 1), size_t(127)}) {
        std::vector<int> logical(count, -1), actual(count, -2), explicit_static(count, -3);
        d.coarse_for_logical_threads(count, [&](size_t i, int tid) {
            require(omp_get_level() == 0, "coarse executor does not fork");
            logical[i] = tid;
        });
        #pragma omp parallel for
        for (size_t i = 0; i < count; ++i) actual[i] = omp_get_thread_num();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i) explicit_static[i] = omp_get_thread_num();
        // The production coarse gate excludes unverified compiler schedules.
        if (d.coarse_dispatch_candidate() && omp_get_thread_limit() >= threads) {
            require(logical == actual, "default scheduling retains logical owners");
            require(logical == explicit_static, "static scheduling retains logical owners");
        }
    }
    d.coarse_serial_ready = false;
    require(!d.coarse_region_is_small(1, 1), "disabled gate");
    require(!Decomp<int32_t>::coarse_dispatch_candidate(), "integer gate");
    omp_set_dynamic(1);
    require(!d.coarse_dispatch_candidate(), "dynamic gate");
    omp_set_dynamic(0);
    int nested_ok = 1;
    #pragma omp parallel reduction(&:nested_ok)
    {
        nested_ok &= !d.coarse_dispatch_candidate();
    }
    require(nested_ok != 0, "nested gate");
}

template <class T, unsigned N>
void release_compression(Decomp<T, N> &d) {
    ::operator delete[](d.total_quant_inds, std::align_val_t(64));
    delete[] d.local_quant_inds;
    delete[] d.local_quant_index;
    ::operator delete[](d.quantizer.unpred_vec, std::align_val_t(64));
    delete[] d.quantizer.local_unpred;
    delete[] d.quantizer.local_unpred_idx;
    d.release_temporary_storage();
}

template <class T, unsigned N>
void roundtrip(const std::array<size_t, N> &dims, int algo, int direction,
               size_t anchor, int pattern) {
    context = "type=" + std::to_string(sizeof(T)) + (std::is_integral_v<T> ? " int" : " fp") +
              " N=" + std::to_string(N) + " P=" + std::to_string(omp_get_max_threads()) +
              " algo=" + std::to_string(algo) + " dir=" + std::to_string(direction) +
              " anchor=" + std::to_string(anchor) + " pattern=" + std::to_string(pattern);
    context += " dims=";
    for (size_t i = 0; i < N; ++i) {
        if (i != 0) context += "x";
        context += std::to_string(dims[i]);
    }
    Falcons::Config conf;
    conf.setDims(dims.begin(), dims.end());
    conf.cmprAlgo = Falcons::ALGO_INTERP;
    conf.openmp = true;
    conf.interpAlgo = algo;
    conf.interpDirection = direction;
    conf.interpAnchorStride = anchor;
    conf.interpAlpha = 1.5;
    conf.interpBeta = 4;
    conf.quantbinCnt = 65536;
    conf.errorBoundMode = Falcons::EB_ABS;
    conf.absErrorBound = std::is_integral_v<T> ? 0.125 : 0.001;
    std::vector<T> original(conf.num);
    for (size_t i = 0; i < original.size(); ++i) {
        const int noise = static_cast<int>((i * 47 + 19) % 127) - 63;
        double value = pattern == 0 ? 1.0 :
            (i % dims[N - 1]) * 0.125 + ((i / dims[N - 1]) % 31) * 0.25 + noise * 0.03125;
        if (pattern == 2 && i % 97 == 41) value += 256;
        original[i] = static_cast<T>(value);
    }
    auto data = original;
    Decomp<T, N> encoder(conf, Falcons::LinearQuantizerOMP<T>(conf.absErrorBound));
    encoder.compress(conf, data.data());
    const size_t threads = encoder.nThreads;
    context += " actualP=" + std::to_string(threads);
    Bytes metadata(1024);
    auto *out = metadata.data();
    encoder.save2(out);
    metadata.resize(out - metadata.data());
    std::vector<std::vector<int16_t>> codes(threads);
    std::vector<Bytes> unpred(threads);
    std::vector<int> code_counts(threads), unpred_counts(threads);
    constexpr size_t code_alignment = 64 / sizeof(int16_t), outlier_alignment = 64 / sizeof(T);
    const size_t code_capacity =
        ((encoder.local_stream_capacity + code_alignment - 1) / code_alignment + 1) * code_alignment;
    const size_t outlier_capacity =
        ((encoder.local_stream_capacity + outlier_alignment - 1) / outlier_alignment) * outlier_alignment;
    for (size_t tid = 0; tid < threads; ++tid) {
        code_counts[tid] = encoder.local_quant_index[tid].value;
        unpred_counts[tid] = encoder.quantizer.local_unpred_idx[tid].value;
        require(code_counts[tid] >= 0 && unpred_counts[tid] >= 0, "nonnegative stream counts");
        require_stream_capacity(tid, code_counts[tid], unpred_counts[tid], code_capacity, outlier_capacity);
        codes[tid].assign(encoder.local_quant_inds[tid],
                          encoder.local_quant_inds[tid] + code_counts[tid]);
        codes[tid].resize(codes[tid].size() + 32, 0);
        unpred[tid].resize(sizeof(int) + unpred_counts[tid] * sizeof(T));
        out = unpred[tid].data();
        encoder.save3(out, static_cast<int>(tid));
        require(out == unpred[tid].data() + unpred[tid].size(), "outlier serialization length");
    }
    release_compression(encoder);

    Decomp<T, N> decoder(conf, Falcons::LinearQuantizerOMP<T>(conf.absErrorBound));
    const auto *cursor = metadata.data();
    size_t remaining = metadata.size();
    decoder.load2(cursor, remaining);
    require(remaining == 0, "metadata consumed");
    decoder.init_local_unpred(static_cast<int>(threads));
    std::vector<int16_t *> pointers(threads);
    for (size_t tid = 0; tid < threads; ++tid) {
        cursor = unpred[tid].data();
        remaining = unpred[tid].size();
        decoder.load3(cursor, remaining, static_cast<int>(tid));
        require(remaining == 0, "outlier stream consumed");
        pointers[tid] = codes[tid].data();
    }
    std::vector<T> recovered(conf.num, T(-7));
    decoder.decompress(conf, pointers.data(), recovered.data());
    for (size_t tid = 0; tid < threads; ++tid) {
        require(decoder.local_quant_index[tid].value == code_counts[tid], "per-thread code cursor");
        require(decoder.quantizer.local_unpred_idx[tid].value == unpred_counts[tid], "per-thread outlier cursor");
    }
    decoder.release_temporary_storage();
    for (size_t tid = 0; tid < threads; ++tid)
        require(pointers[tid] == codes[tid].data(), "decoded quantization codes remain borrowed");
    for (size_t i = 0; i < recovered.size(); ++i) {
        require_reconstruction(i, original[i], data[i], recovered[i], conf.absErrorBound);
    }
    ++cases;
}

void ownership_checks() {
    context = "OMP decompression ownership";
    Falcons::Config conf(129);
    conf.cmprAlgo = Falcons::ALGO_INTERP;
    conf.interpAnchorStride = 0;
    conf.absErrorBound = 0.001;
    Decomp<float, 1> decoder(conf, Falcons::LinearQuantizerOMP<float>(conf.absErrorBound));
    decoder.init_local_unpred(1);
    unsigned char serialized[sizeof(int) + 2 * sizeof(float)];
    auto *out = serialized;
    Falcons::write<int>(2, out);
    Falcons::write<float>(42.0f, out);
    Falcons::write<float>(84.0f, out);
    const auto *in = serialized;
    size_t remaining = sizeof(serialized);
    decoder.load3(in, remaining, 0);
    require(decoder.quantizer.local_unpred[0] != nullptr, "decoded outlier allocated");
    require(decoder.quantizer.recover_unpred2(0) == 42.0f, "first outlier recovered");
    auto *const old_outliers = decoder.quantizer.local_unpred[0];
    const int old_cursor = decoder.quantizer.local_unpred_idx[0].value;
    unsigned char truncated[sizeof(int) + sizeof(float)];
    out = truncated;
    Falcons::write<int>(2, out);
    Falcons::write<float>(-1.0f, out);
    const auto *truncated_in = truncated;
    size_t truncated_remaining = sizeof(truncated);
    bool rejected = false;
    try { decoder.load3(truncated_in, truncated_remaining, 0); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && decoder.quantizer.local_unpred[0] == old_outliers &&
            decoder.quantizer.decoded_unpred_owner[0].get() == old_outliers &&
            decoder.quantizer.local_unpred_idx[0].value == old_cursor,
            "truncated reload preserves outlier state");
    require(truncated_in == truncated && truncated_remaining == sizeof(truncated),
            "truncated reload preserves input cursor");
    require(decoder.quantizer.recover_unpred2(0) == 84.0f,
            "old outliers remain usable after truncated reload");
    int16_t code = 7;
    int16_t *borrowed = &code;
    rejected = false;
    try { decoder.decompress(conf, &borrowed, nullptr); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && !decoder.quantizer.local_unpred && !decoder.local_quant_index,
            "failed decompression releases outliers and cursors");
    require(borrowed == &code && code == 7, "failure does not free borrowed codes");
}

#ifdef __AVX2__
template <class T>
void avx_checks() {
    context = "AVX2 helpers type=" + std::to_string(sizeof(T));
    constexpr size_t lanes = 32 / sizeof(T);
    std::vector<T> src(2 * lanes - 1), dst(2 * lanes - 1, T(-777));
    std::array<T, lanes> selected;
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<T>(i + 1);
    if constexpr (std::is_same_v<T, float>) {
        const auto value = Falcons::avx2_load_stride2(src.data());
        _mm256_storeu_ps(selected.data(), value);
        Falcons::avx2_store_stride2(dst.data(), value);
    } else {
        const auto value = Falcons::avx2_load_stride2(src.data());
        _mm256_storeu_pd(selected.data(), value);
        Falcons::avx2_store_stride2(dst.data(), value);
    }
    for (size_t i = 0; i < src.size(); ++i) {
        require(dst[i] == (i % 2 == 0 ? src[i] : T(-777)), "masked store preserves interleaved lanes");
        if (i % 2 == 0) require(selected[i / 2] == src[i], "stride2 lane order");
    }
    for (size_t count : {size_t(1), lanes - 1, lanes, lanes + 1, 2 * lanes + 1}) {
        std::vector<T> input(2 * count - 1), packed(count);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<T>(3 * i + 1);
        Falcons::packing_detail::copy_stride2(packed.data(), input.data(), count);
        for (size_t i = 0; i < count; ++i) require(packed[i] == input[2 * i], "packing tail/lane order");
    }
    const unsigned saved_csr = _mm_getcsr();
    _mm_setcsr(0x1f80u);
    const bool default_fp_ok = Decomp<T>::coarse_fp_environment_ok();
    bool accepted, rejected, ftz_rejected;
    if constexpr (std::is_same_v<T, float>) {
        const auto eb = _mm256_set1_ps(0.125f), neb = _mm256_set1_ps(-0.125f);
        const auto bound = Falcons::avx2_zero_code_bound<true>(eb, neb, 0.25, 4.0, 32768, true);
        accepted = Falcons::is_avx2_zero_code_batch(eb, bound);
        rejected = !Falcons::is_avx2_zero_code_batch(_mm256_set1_ps(0.25f), bound);
        _mm_setcsr(0x9fc0u);
        const auto disabled = Falcons::avx2_zero_code_bound<true>(eb, neb, 0.25, 4.0, 32768, true);
        ftz_rejected = !Falcons::is_avx2_zero_code_batch(_mm256_setzero_ps(), disabled);
    } else {
        const auto eb = _mm256_set1_pd(0.125), neb = _mm256_set1_pd(-0.125);
        const auto bound = Falcons::avx2_zero_code_bound<true>(eb, neb, 0.25, 4.0, 32768, true);
        accepted = Falcons::is_avx2_zero_code_batch(eb, bound);
        rejected = !Falcons::is_avx2_zero_code_batch(_mm256_set1_pd(0.25), bound);
        _mm_setcsr(0x9fc0u);
        const auto disabled = Falcons::avx2_zero_code_bound<true>(eb, neb, 0.25, 4.0, 32768, true);
        ftz_rejected = !Falcons::is_avx2_zero_code_batch(_mm256_setzero_pd(), disabled);
    }
    const bool coarse_ftz_rejected = !Decomp<T>::coarse_fp_environment_ok();
    _mm_setcsr(saved_csr);
    require(accepted && rejected, "zero-code bound");
    require(ftz_rejected, "FTZ/DAZ disables zero-code shortcut");
    require(default_fp_ok && coarse_ftz_rejected, "coarse FP environment gate");
}
#endif
} // namespace

int main(int argc, char **argv) {
    try {
        int only_threads = 0;
        if (argc == 3 && std::string(argv[1]) == "--threads") only_threads = std::stoi(argv[2]);
        else if (argc != 1) throw std::runtime_error("usage: v11_omp_test [--threads 1|2|4|8|32]");
        if (only_threads && only_threads != 1 && only_threads != 2 && only_threads != 4 &&
            only_threads != 8 && only_threads != 32)
            throw std::runtime_error("unsupported thread count");
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        scalar_capacity_checks();
        simd_capacity_checks();
#ifdef __AVX2__
        avx_checks<float>();
        avx_checks<double>();
#endif
        for (int threads : {1, 2, 4, 8, 32}) {
            if (only_threads && only_threads != threads) continue;
            omp_set_num_threads(threads);
            partition_checks(threads);
            ownership_checks();
            for (int algo : {0, 1}) {
                for (int direction : {0, 1, 2, 3, 4, 5}) {
                    roundtrip<float, 3>({33, 35, 67}, algo, direction, 0, 0);
                    roundtrip<double, 3>({34, 37, 65}, algo, direction, 8, 1);
                    roundtrip<float, 3>({35, 37, 39}, algo, direction, 8, 2);
                    roundtrip<int32_t, 3>({33, 35, 37}, algo, direction, 0, 1);
                    if (threads == 32) {
                        roundtrip<float, 3>({36, 35, 37}, algo, direction, 0, 1);
                        roundtrip<double, 3>({36, 35, 37}, algo, direction, 0, 1);
                    }
                }
                roundtrip<float, 2>({35, 67}, algo, 0, 0, 2);
                roundtrip<double, 2>({34, 65}, algo, 1, 8, 0);
                roundtrip<int32_t, 2>({35, 37}, algo, 0, 0, 1);
                roundtrip<int32_t, 2>({35, 37}, algo, 1, 8, 1);
                roundtrip<int32_t, 1>({137}, algo, 0, 0, 1);
                if (threads == 32) {
                    for (int direction : {0, 1}) {
                        for (size_t anchor : {size_t(0), size_t(8)}) {
                            roundtrip<float, 2>({34, 65}, algo, direction, anchor, 1);
                            roundtrip<double, 2>({34, 65}, algo, direction, anchor, 0);
                        }
                        roundtrip<float, 2>({2, 257}, algo, direction, 0, 1);
                    }
                    roundtrip<double, 2>({2, 257}, algo, 1, 0, 1);
                    for (int direction : {0, 5}) {
                        for (size_t anchor : {size_t(0), size_t(8)}) {
                            roundtrip<float, 3>({2, 33, 65}, algo, direction, anchor, 1);
                            roundtrip<double, 3>({2, 33, 65}, algo, direction, anchor, 1);
                        }
                    }
                }
                if (threads <= 8) roundtrip<float, 1>({137}, algo, 0, 0, 1);
                if (threads == 1) {
                    roundtrip<float, 3>({2, 2, 2}, algo, 0, 0, 0);
                    roundtrip<double, 3>({3, 4, 5}, algo, 5, 0, 2);
                }
            }
            require(omp_get_max_threads() == threads, "caller thread setting preserved");
        }
        std::printf("PASS: %zu same-build cases, %zu checks\n", cases, checks);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s: %s\n", context.c_str(), error.what());
        return 1;
    }
}
#endif
