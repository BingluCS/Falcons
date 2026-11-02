// Standalone synthetic correctness checks for the parent agent's server runs.
// Use the current source and generated include directories, link Zstd, and
// optionally enable OpenMP and AVX2/SVE2. No timing or legacy-stream checks.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "Falcons/api/impl/FalconsAlgoInterp.hpp"

namespace {
using Falcons::Config;
using Falcons::uchar;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
void check_minmax(const T *data, size_t begin, size_t end) {
    T expected_min, expected_max, actual_min, actual_max;
    Falcons::scalar_minmax(data, begin, end, expected_min, expected_max);
#ifdef __ARM_FEATURE_SVE2
    Falcons::sve2_minmax(data, begin, end, actual_min, actual_max);
#elif defined(__AVX2__)
    Falcons::avx2_minmax(data, begin, end, actual_min, actual_max);
#else
    Falcons::scalar_minmax(data, begin, end, actual_min, actual_max);
#endif
    require(actual_min == expected_min && actual_max == expected_max, "min/max mismatch");
    const T expected_range = expected_max - expected_min;
    require(Falcons::data_range(data + begin, end - begin, false) == expected_range, "serial range mismatch");
    require(Falcons::data_range(data + begin, end - begin, true) == expected_range, "parallel/fallback range mismatch");
    require(Falcons::data_range(data + begin, end - begin) == expected_range, "default range mismatch");
}

template <class T>
void check_ranges() {
    require(Falcons::data_range<T>(nullptr, 0, false) == T(0), "empty serial range mismatch");
    require(Falcons::data_range<T>(nullptr, 0, true) == T(0), "empty parallel range mismatch");
    for (size_t begin = 0; begin < 8; ++begin) {
        for (size_t count : {size_t(1), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32),
                             size_t(33), size_t(63), size_t(64), size_t(65), size_t(137)}) {
            std::vector<T> data(begin + count);
            for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<T>((i * 17) % 51);
            data[begin + count / 3] = T(0);
            data.back() = T(63);
            check_minmax(data.data(), begin, data.size());
        }
    }
    if constexpr (std::is_floating_point_v<T>) {
        std::vector<T> data(137);
        for (size_t i = 0; i < data.size(); ++i) data[i] = T(static_cast<int>(i % 29) - 14) / T(4);
        check_minmax(data.data(), 0, data.size());
        std::fill(data.begin(), data.end(), -T(0));
        data[0] = T(0);
        check_minmax(data.data(), 0, data.size());
        data.front() = -std::numeric_limits<T>::infinity();
        data.back() = std::numeric_limits<T>::infinity();
        check_minmax(data.data(), 0, data.size());
        std::fill(data.begin(), data.end(), std::numeric_limits<T>::quiet_NaN());
        require(std::isnan(Falcons::data_range(data.data(), data.size(), false)), "all-NaN range mismatch");
    }
}

template <class T>
void check_bounds() {
    std::array<T, 37> data;
    for (size_t i = 0; i < data.size(); ++i) data[i] = T(static_cast<int>(i % 23) - 7);
    for (bool parallel : {false, true}) {
        for (int mode = Falcons::EB_ABS; mode <= Falcons::EB_ABS_OR_REL; ++mode) {
            for (T supplied_range : {T(0), T(2), T(-1)}) {
                Config conf(data.size());
                conf.openmp = parallel;
                conf.errorBoundMode = mode;
                conf.absErrorBound = 0.125;
                conf.relErrorBound = 0.01;
                conf.psnrErrorBound = 60;
                conf.l2normErrorBound = 0.5;
                const double range = supplied_range > 0 ? supplied_range : T(22);
                double expected = conf.absErrorBound;
                switch (mode) {
                    case Falcons::EB_REL: expected = conf.relErrorBound * range; break;
                    case Falcons::EB_PSNR:
                        expected = Falcons::computeABSErrBoundFromPSNR(conf.psnrErrorBound, 0.99, range);
                        break;
                    case Falcons::EB_L2NORM:
                        expected = std::sqrt(3.0 / conf.num) * conf.l2normErrorBound;
                        break;
                    case Falcons::EB_ABS_AND_REL: expected = std::min(expected, conf.relErrorBound * range); break;
                    case Falcons::EB_ABS_OR_REL: expected = std::max(expected, conf.relErrorBound * range); break;
                }
                Falcons::calAbsErrorBound(conf, data.data(), supplied_range);
                require(conf.errorBoundMode == Falcons::EB_ABS && conf.absErrorBound == expected, "error bound mismatch");
            }
        }
    }
}

template <class T>
std::vector<std::vector<T>> make_samples(size_t num) {
    std::vector<std::vector<T>> samples(7);
    for (size_t k = 0; k < samples.size(); ++k) {
        // Trailing elements exercise worker-buffer grow/shrink, outside conf.num.
        samples[k].resize(num + k % 3);
        for (size_t i = 0; i < samples[k].size(); ++i) {
            double value = std::sin(i * 0.13) + std::cos(i * 0.007) + i * 0.001;
            if (k % 3 == 1) value = static_cast<double>((i * 7919 + k * 104729) % 200003) / 1024 - 98;
            if (k % 3 == 2) value = -3.5;
            samples[k][i] = static_cast<T>(value);
        }
    }
    return samples;
}

template <class T, Falcons::uint N>
double score_from_current_symbols(const std::vector<std::vector<T>> &samples, const Config &conf) {
    std::vector<T> unpredictable;
    std::vector<int16_t> symbols;
    for (const auto &sample : samples) {
        auto copy = sample;
        auto decomposition = Falcons::make_decomposition_interpolation<Falcons::TUNING::ENABLED, T, N>(
            conf, Falcons::LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2));
        auto [indices, count] = decomposition.compress(conf, copy.data());
        std::unique_ptr<int16_t[]> owned_indices(indices);
        const auto &raw = decomposition.test_unpred();
        unpredictable.insert(unpredictable.end(), raw.begin(), raw.end());
        symbols.insert(symbols.end(), indices, indices + count);
        for (size_t i = 0; i < conf.num; ++i) {
            const long double error = std::fabs(static_cast<long double>(copy[i]) - sample[i]);
            require(error <= conf.absErrorBound, "sample reconstruction exceeds error bound");
        }
    }
    const size_t raw_bytes = unpredictable.size() * sizeof(T);
    const size_t symbol_bytes = symbols.size() * sizeof(int16_t);
    std::vector<uchar> payload(std::max<size_t>(1, raw_bytes + symbol_bytes));
    if (raw_bytes != 0) std::memcpy(payload.data(), unpredictable.data(), raw_bytes);
    if (symbol_bytes != 0) std::memcpy(payload.data() + raw_bytes, symbols.data(), symbol_bytes);
    std::vector<uchar> compressed(ZSTD_compressBound(raw_bytes + symbol_bytes));
    const size_t size = ZSTD_compress(compressed.data(), compressed.size(), payload.data(), raw_bytes + symbol_bytes, 3);
    require(!ZSTD_isError(size), "reference scoring failed");
    return conf.num * samples.size() * sizeof(T) * 1.0 / (sizeof(size_t) + size);
}

template <class T, Falcons::uint N>
void check_candidates(bool parallel) {
    constexpr size_t edge = N == 1 ? 65 : (N == 2 ? 17 : 9);
    std::array<size_t, N> dims;
    dims.fill(edge);
    Config base;
    base.setDims(dims.begin(), dims.end());
    base.openmp = parallel;
    base.absErrorBound = 0.001;
    const auto samples = make_samples<T>(base.num);
    const auto saved = samples;
    const size_t capacity = sizeof(size_t) + ZSTD_compressBound(base.num * samples.size() * (sizeof(T) + sizeof(int16_t)));
    std::vector<uchar> compressed(capacity);
    for (int mode = 0; mode < 2; ++mode) {
        base.interpAnchorStride = mode == 0 ? 8 : 0;
        base.quantbinCnt = mode == 0 ? 65536 : 64;
        std::vector<Config> candidates(N <= 2 ? 8 : 4, base);
        std::vector<double> expected(candidates.size()), single(candidates.size()), batch(candidates.size());
        for (size_t c = 0; c < candidates.size(); ++c) {
            auto &conf = candidates[c];
            conf.interpAlgo = c % 4 / 2 == 0 ? Falcons::INTERP_ALGO_LINEAR : Falcons::INTERP_ALGO_CUBIC;
            conf.interpDirection = c % 2 == 0 ? 0 : Falcons::factorial(N) - 1;
            conf.interpAlpha = N <= 2 && c < 4 ? 1.0 : 1.25;
            conf.interpBeta = N <= 2 && c < 4 ? 1.0 : 2.0;
            expected[c] = score_from_current_symbols<T, N>(samples, conf);
            single[c] = Falcons::interp_compress_test<T, N>(samples, conf, edge - 1, compressed.data(), capacity);
        }
        Falcons::interp_compress_test_batch<T, N>(samples, candidates.data(), candidates.size(), edge - 1, batch.data());
        require(single == expected && batch == expected, "candidate scores disagree with current symbols/Zstd");
        require(samples == saved, "scoring modified source samples");
        std::fill(batch.begin(), batch.end(), -1);
        Falcons::interp_compress_test_batch<T, N>(samples, candidates.data(), candidates.size(), edge - 1, batch.data());
        require(batch == expected && samples == saved, "repeated scoring changed results or samples");
    }
    double untouched = -7;
    Falcons::interp_compress_test_batch<T, N>(samples, nullptr, 0, edge - 1, &untouched);
    Falcons::interp_compress_test_batch<T, N>(samples, nullptr, -1, edge - 1, &untouched);
    require(untouched == -7, "empty candidate set wrote a ratio");
    double empty_ratio = -1;
    Falcons::interp_compress_test_batch<T, N>({}, &base, 1, edge - 1, &empty_ratio);
    require(empty_ratio == 0, "empty samples produced a nonzero ratio");
    bool rejected = false;
    try {
        Falcons::interp_compress_test<T, N>(samples, base, edge - 1, compressed.data(), sizeof(size_t));
    } catch (const std::length_error &) {
        rejected = true;
    }
    require(rejected, "undersized scoring buffer was accepted");
}
}  // namespace

int main() {
    try {
        require(std::getenv("FALCONS_NOZSTD") == nullptr, "FALCONS_NOZSTD must be unset");
        check_ranges<float>();
        check_ranges<double>();
        check_ranges<int8_t>();
        check_ranges<uint8_t>();
        check_ranges<int16_t>();
        check_ranges<uint16_t>();
        check_ranges<int32_t>();
        check_ranges<uint32_t>();
        check_ranges<int64_t>();
        check_ranges<uint64_t>();
        check_bounds<float>();
        check_bounds<double>();
        constexpr uint64_t high = uint64_t(1) << 63;
        require(Falcons::statistic_difference(high + 99, high) == 99, "unsigned difference regression");
        require(Falcons::statistic_difference(high, high + 99) == -99, "unsigned difference sign regression");
        require(Falcons::statistic_difference(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min())
                    == std::ldexp(1.0, 64), "signed difference regression");
        for (bool parallel : {false, true}) {
            check_candidates<float, 1>(parallel);
            check_candidates<float, 2>(parallel);
            check_candidates<float, 3>(parallel);
            check_candidates<float, 4>(parallel);
            check_candidates<double, 1>(parallel);
            check_candidates<double, 2>(parallel);
            check_candidates<double, 3>(parallel);
            check_candidates<double, 4>(parallel);
        }
        std::cout << "V1.1 tuning/range correctness checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
