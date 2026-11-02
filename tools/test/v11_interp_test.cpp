// Standalone correctness checks, not a benchmark. Run only on the test server.
// c++ -std=c++17 -O2 -mavx2 -mfma -Iinclude -Ibuild/include \
//     tools/test/v11_interp_test.cpp -o /tmp/v11_interp_test
// Also build without ISA flags for scalar, and with both
// Falcons_SERIAL_DECOMP_ORIGINAL_{STORES,PACKING}=0 for the opt-in alternatives.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "Falcons/quantizer/LinearQuantizer.hpp"
#include "Falcons/decomposition/InterpolationDecomposition.hpp"

namespace {
std::size_t cases = 0;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T> class EndpointArray {
#if defined(__unix__) || defined(__APPLE__)
    void *mapping;
    std::size_t page;
#else
    std::unique_ptr<T[]> storage;
#endif
public:
    T *data;
    explicit EndpointArray(std::size_t count) {
#if defined(__unix__) || defined(__APPLE__)
        const auto page_size = sysconf(_SC_PAGESIZE);
        require(page_size > 0, "page size");
        page = static_cast<std::size_t>(page_size);
        require(count <= page / sizeof(T), "guard-page extent");
        mapping = mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(mapping != MAP_FAILED, "mmap");
        if (mprotect(static_cast<char *>(mapping) + page, page, PROT_READ | PROT_WRITE)) {
            munmap(mapping, 3 * page);
            throw std::runtime_error("mprotect");
        }
        data = reinterpret_cast<T *>(static_cast<char *>(mapping) + 2 * page) - count;
#else
        storage.reset(new T[count]);
        data = storage.get();
#endif
    }
    ~EndpointArray() {
#if defined(__unix__) || defined(__APPLE__)
        munmap(mapping, 3 * page);
#endif
    }
    EndpointArray(const EndpointArray &) = delete;
    EndpointArray &operator=(const EndpointArray &) = delete;
};

template <class T> void packing_checks() {
    for (std::size_t count = 0; count <= 67; ++count) {
        const auto span = count ? 2 * count - 1 : 0;
        EndpointArray<T> input(span);
        for (std::size_t i = 0; i < span; ++i) input.data[i] = T(i + 11);
        std::vector<T> output(count + 2, T(99));
        Falcons::packing_detail::copy_stride2(output.data() + 1, input.data, count);
        require(output.front() == T(99) && output.back() == T(99), "packing output boundary");
        for (std::size_t i = 0; i < count; ++i)
            require(output[i + 1] == input.data[2 * i], "packing lane order");
        ++cases;
    }
#ifdef __AVX2__
    constexpr std::size_t lanes = 32 / sizeof(T), span = 2 * lanes - 1;
    EndpointArray<T> input(span);
    for (std::size_t i = 0; i < span; ++i) input.data[i] = T(i + 21);
    std::array<T, lanes> selected;
    const auto value = Falcons::avx2_load_stride2(input.data);
    if constexpr (std::is_same_v<T, float>) _mm256_storeu_ps(selected.data(), value);
    else _mm256_storeu_pd(selected.data(), value);
    for (std::size_t i = 0; i < lanes; ++i)
        require(selected[i] == T(2 * i + 21), "stride-2 load order");
    std::fill_n(input.data, span, T(-1));
    Falcons::avx2_store_stride2(input.data, value);
    for (std::size_t i = 0; i < span; ++i)
        require(input.data[i] == (i % 2 ? T(-1) : T(i + 21)), "stride-2 store mask");
    ++cases;
#endif
}

#ifdef __AVX2__
void zero_code_checks() {
    const auto bound_d = Falcons::avx2_zero_code_bound<true>(
        _mm256_set1_pd(0.125), _mm256_set1_pd(-0.125), 0.25, 4.0, 32768, true);
    const auto bound_f = Falcons::avx2_zero_code_bound<true>(
        _mm256_set1_ps(0.125f), _mm256_set1_ps(-0.125f), 0.25, 4.0, 32768, true);
    require(Falcons::is_avx2_zero_code_batch(_mm256_set1_pd(-0.125), bound_d), "double zero-code tie");
    require(Falcons::is_avx2_zero_code_batch(_mm256_set1_ps(0.125f), bound_f), "float zero-code tie");
    require(!Falcons::is_avx2_zero_code_batch(
                _mm256_set1_pd(std::nextafter(0.125, 1.0)), bound_d), "double nonzero-code boundary");
    require(!Falcons::is_avx2_zero_code_batch(
                _mm256_set1_ps(std::nextafter(0.125f, 1.0f)), bound_f), "float nonzero-code boundary");
    const auto disabled = Falcons::avx2_zero_code_bound<true>(
        _mm256_set1_pd(0.125), _mm256_set1_pd(-0.125), 0.25, 4.0, 32768, false);
    require(!Falcons::is_avx2_zero_code_batch(_mm256_setzero_pd(), disabled), "zero-code tail fallback");
    const auto diff = _mm256_setr_pd(-0.125, 0.125, -0.0, 0.0);
    const auto pred = _mm256_setr_pd(-0.0, -0.0, -1.0, 1.0);
    std::array<double, 4> reconstructed;
    _mm256_storeu_pd(reconstructed.data(), Falcons::avx2_zero_code_reconstruct(diff, pred, _mm256_set1_pd(0.25)));
    require(reconstructed[0] == 0 && std::signbit(reconstructed[0]) &&
                reconstructed[1] == 0 && !std::signbit(reconstructed[1]) &&
                reconstructed[2] == -1 && reconstructed[3] == 1, "zero-code reconstruction");
    const unsigned saved = _mm_getcsr();
    _mm_setcsr(saved | _MM_FLUSH_ZERO_MASK);
    const auto ftz_bound = Falcons::avx2_zero_code_bound<true>(
        _mm256_set1_pd(0.125), _mm256_set1_pd(-0.125), 0.25, 4.0, 32768, true);
    const bool rejects_ftz = !Falcons::is_avx2_zero_code_batch(_mm256_setzero_pd(), ftz_bound);
    _mm_setcsr(saved);
    require(rejects_ftz, "zero-code FTZ fallback");
    ++cases;
}
#endif

template <class T> std::vector<T> fixture(std::size_t size, int pattern) {
    std::vector<T> data(size);
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (std::is_integral_v<T>) {
            constexpr T lo = std::numeric_limits<T>::min(), hi = std::numeric_limits<T>::max();
            const T values[] = {lo, hi, T(0), T(1), T(lo + 1), T(hi - 1), T(7)};
            data[i] = values[i % 7];
        } else {
            if (pattern == 0) data[i] = T(0);
            else if (pattern == 1) data[i] = T(std::sin(i * 0.03125) + std::cos(i * 0.0078125));
            else data[i] = T((int((i * 47) % 127) - 63) * 0.03125 + (i % 17 == 0 ? 10000 : 0));
        }
    }
    return data;
}

template <class T> long double distance(T lhs, T rhs) {
    if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        return lhs < rhs ? U(rhs) - U(lhs) : U(lhs) - U(rhs);
    } else {
        return std::abs(static_cast<long double>(lhs) - static_cast<long double>(rhs));
    }
}

template <class T, Falcons::uint N, Falcons::TUNING Mode>
void round_trip(const std::array<std::size_t, N> &dims, int algo, int direction,
                std::size_t anchor, int pattern) {
    Falcons::Config config;
    config.setDims(dims.begin(), dims.end());
    require(config.N == N, "test fixture dimension");
    config.interpAlgo = algo;
    config.interpDirection = direction;
    config.interpAnchorStride = anchor;
    config.interpAlpha = 1.25;
    config.interpBeta = 2;
    config.absErrorBound = 0.125;
    auto data = fixture<T>(config.num, pattern);
    const auto original = data;
    using Decomposition = Falcons::InterpolationDecomposition<Mode, T, N, Falcons::LinearQuantizer<T>>;
    Decomposition encoder(config, Falcons::LinearQuantizer<T>(config.absErrorBound));
    auto [raw_codes, count] = encoder.compress(config, data.data());
    std::unique_ptr<int16_t[]> codes(raw_codes);
    require(count <= config.num && (anchor || count == config.num), "quantization count");
    std::vector<unsigned char> metadata(config.num * sizeof(T) + 1024);
    auto *write = metadata.data();
    encoder.save(write);
    const auto *read = metadata.data();
    std::size_t remaining = static_cast<std::size_t>(write - metadata.data());
    Decomposition decoder(config, Falcons::LinearQuantizer<T>(config.absErrorBound));
    decoder.load(read, remaining);
    require(remaining == 0, "same-version metadata consumed");
    std::vector<T> decoded(config.num);
    decoder.decompress(config, codes.get(), decoded.data());
    for (std::size_t i = 0; i < config.num; ++i)
        require(std::isfinite(static_cast<long double>(decoded[i])) &&
                    distance(original[i], decoded[i]) <= config.absErrorBound,
                "round-trip bound N=" + std::to_string(N) + " algo=" + std::to_string(algo) +
                    " dir=" + std::to_string(direction) + " i=" + std::to_string(i));
    ++cases;
}

template <class T, Falcons::TUNING Mode> void field_checks() {
    for (int algo : {0, 1}) {
        for (std::size_t n : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 15u, 16u, 17u, 18u, 31u, 32u, 33u, 65u, 66u})
            for (int pattern : {0, 1, 2}) round_trip<T, 1, Mode>({n}, algo, 0, 0, pattern);
        for (std::size_t anchor : {0u, 8u}) {
            for (int direction : {0, 1}) {
                round_trip<T, 2, Mode>({2, 3}, algo, direction, anchor, 2);
                round_trip<T, 2, Mode>({17, 33}, algo, direction, anchor, 1);
                round_trip<T, 2, Mode>({18, 34}, algo, direction, anchor, 0);
            }
            for (int direction = 0; direction < 6; ++direction) {
                if constexpr (Mode == Falcons::TUNING::DISABLED)
                    if (direction != 0 && direction != 5) continue;
                round_trip<T, 3, Mode>({2, 3, 4}, algo, direction, anchor, 2);
                round_trip<T, 3, Mode>({17, 19, 33}, algo, direction, anchor, 1);
                round_trip<T, 3, Mode>({18, 19, 34}, algo, direction, anchor, 0);
            }
            for (int direction = 0; direction < 24; ++direction)
                round_trip<T, 4, Mode>({3, 5, 7, 17}, algo, direction, anchor, 2);
        }
    }
}

#if Falcons_TUNING_SIMD_AVAILABLE
template <class T> void tuning_checks() {
    const Falcons::tuning_linear_detail::Parameters p{0.125, 0.25, 4.0, 32768};
    require(p.eligible(), "default tuning environment");
    for (std::size_t point_stride : {1u, 2u, 5u}) {
        constexpr std::size_t neighbor_stride = 19;
        auto data = fixture<T>(6 * neighbor_stride + 3 * point_stride + 1, 1);
        const T *line = data.data() + 3 * neighbor_stride;
        for (bool cubic : {false, true}) {
            Falcons::tuning_linear_detail::Result<T> result;
            const bool processed = cubic
                ? Falcons::tuning_linear_detail::cubic4(line, point_stride, neighbor_stride, p, result)
                : Falcons::tuning_linear_detail::linear4(line, point_stride, neighbor_stride, p, result);
            require(processed, "finite tuning batch");
            require(result.accepted != 0, "tuning accepted lanes");
            for (std::size_t lane = 0; lane < 4; ++lane) {
                if (!(result.accepted & (1u << lane))) continue;
                require(result.codes[lane] > -p.radius && result.codes[lane] < p.radius &&
                            distance(result.reconstructed[lane], line[lane * point_stride]) <= p.bound,
                        "tuning reconstruction bound");
            }
            ++cases;
        }
    }
    const unsigned saved = _mm_getcsr();
    _mm_setcsr(saved | _MM_FLUSH_ZERO_MASK);
    const bool rejects_ftz = !p.eligible();
    _mm_setcsr(saved);
    require(rejects_ftz, "tuning FTZ fallback");
}
#endif

} // namespace

int main() {
    try {
        packing_checks<float>();
        packing_checks<double>();
#ifdef __AVX2__
        zero_code_checks();
#endif
        field_checks<float, Falcons::TUNING::DISABLED>();
        field_checks<double, Falcons::TUNING::DISABLED>();
        field_checks<float, Falcons::TUNING::ENABLED>();
        field_checks<double, Falcons::TUNING::ENABLED>();
        for (int algo : {0, 1}) {
            round_trip<int32_t, 1, Falcons::TUNING::DISABLED>({33}, algo, 0, 0, 0);
            round_trip<int64_t, 2, Falcons::TUNING::ENABLED>({9, 17}, algo, 1, 0, 0);
            round_trip<uint64_t, 3, Falcons::TUNING::ENABLED>({3, 5, 17}, algo, 5, 0, 0);
        }
#if Falcons_TUNING_SIMD_AVAILABLE
        tuning_checks<float>();
        tuning_checks<double>();
#endif
        std::printf("PASS v11 interpolation: %zu cases\n", cases);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL v11 interpolation: %s\n", error.what());
        return 1;
    }
}
