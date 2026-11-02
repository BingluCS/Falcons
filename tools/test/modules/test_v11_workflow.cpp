#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "Falcons/api/falcons.hpp"
#include "gtest/gtest.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

template <typename T>
std::vector<T> make_values(size_t count) {
    std::vector<T> values(count);
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_integral_v<T>) {
            const auto offset = static_cast<int64_t>((i * 31 + i / 97) % 20001) - 10000;
            if constexpr (std::is_unsigned_v<T>)
                values[i] = static_cast<T>((uint64_t{1} << 60) + offset);
            else
                values[i] = static_cast<T>(500000 + offset);
        } else {
            const double x = static_cast<double>(i);
            values[i] = static_cast<T>(std::sin(x * 0.001) * 10.0 +
                                       std::cos(x * 0.00031) +
                                       ((i % 4093) == 0 ? 12.0 : 0.0));
        }
    }
    return values;
}

template <typename T>
void verify_roundtrip(Falcons::Config config, double absolute_bound) {
    auto original = make_values<T>(config.num);
    auto input = original;
    config.absErrorBound = absolute_bound;
    config.errorBoundMode = Falcons::EB_ABS;
    size_t compressed_size = 0;
    std::unique_ptr<char[]> compressed(Falcons_compress(config, input.data(), compressed_size));
    ASSERT_GT(compressed_size, 0u);
    Falcons::Config decoded_config;
    std::unique_ptr<T[]> decoded(Falcons_decompress<T>(decoded_config, compressed.get(), compressed_size));
    ASSERT_EQ(decoded_config.dims, config.dims);
    ASSERT_EQ(decoded_config.num, config.num);
    for (size_t i = 0; i < config.num; ++i) {
        const long double error = std::abs(static_cast<long double>(decoded[i]) -
                                           static_cast<long double>(original[i]));
        ASSERT_LE(error, static_cast<long double>(absolute_bound)) << "index=" << i;
    }
}

template <typename T>
void verify_interpolation(bool parallel) {
    for (auto algorithm : {Falcons::INTERP_ALGO_LINEAR, Falcons::INTERP_ALGO_CUBIC}) {
        for (int direction : {0, 1, 2, 3, 4, 5}) {
            Falcons::Config config(36, 35, 37);
            config.cmprAlgo = Falcons::ALGO_INTERP;
            config.interpAlgo = algorithm;
            config.interpDirection = direction;
            config.openmp = parallel;
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel << " algorithm=" << algorithm
                                             << " direction=" << direction);
            verify_roundtrip<T>(config, 0.01);
        }
    }
}

}  // namespace

TEST(V11WorkflowTest, SerialFloatingInterpolation) {
    verify_interpolation<float>(false);
    verify_interpolation<double>(false);
}

TEST(V11WorkflowTest, SerialIntegerInterpolation) {
    for (auto algorithm : {Falcons::INTERP_ALGO_LINEAR, Falcons::INTERP_ALGO_CUBIC}) {
        Falcons::Config config(17, 19, 21);
        config.cmprAlgo = Falcons::ALGO_INTERP;
        config.interpAlgo = algorithm;
        SCOPED_TRACE(::testing::Message() << "algorithm=" << algorithm);
        verify_roundtrip<int32_t>(config, 200.0);
        verify_roundtrip<uint64_t>(config, 200.0);
    }
}

TEST(V11WorkflowTest, AutoTuningRoundTrip) {
    Falcons::Config config(96, 97, 98);
    config.cmprAlgo = Falcons::ALGO_INTERP_LORENZO;
    verify_roundtrip<float>(config, 0.01);
}

TEST(V11WorkflowTest, CollapsedDimensionsRoundTrip) {
    Falcons::Config config(1, 257);
    ASSERT_EQ(config.N, 1);
    config.cmprAlgo = Falcons::ALGO_INTERP;
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(32);
#endif
    for (auto algorithm : {Falcons::INTERP_ALGO_LINEAR, Falcons::INTERP_ALGO_CUBIC}) {
        config.interpAlgo = algorithm;
        for (bool parallel : {false, true}) {
            config.openmp = parallel;
            SCOPED_TRACE(::testing::Message() << "parallel=" << parallel << " algorithm=" << algorithm);
            verify_roundtrip<float>(config, 0.01);
            verify_roundtrip<double>(config, 0.01);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
}

TEST(V11WorkflowTest, OpenMPFloatingInterpolation) {
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(32);
    verify_interpolation<float>(true);
    verify_interpolation<double>(true);
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#else
    GTEST_SKIP() << "OpenMP is unavailable in this build";
#endif
}

TEST(V11WorkflowTest, OpenMPIntegerCubicRegression) {
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(32);
    Falcons::Config config(33, 35, 37);
    config.cmprAlgo = Falcons::ALGO_INTERP;
    config.openmp = true;
    config.interpAlgo = Falcons::INTERP_ALGO_CUBIC;
    config.interpDirection = 5;
    config.interpAnchorStride = 0;
    config.interpAlpha = 1.5;
    config.interpBeta = 4;
    config.quantbinCnt = 65536;
    config.errorBoundMode = Falcons::EB_ABS;
    config.absErrorBound = 0.125;
    std::vector<int32_t> original(config.num);
    for (size_t i = 0; i < original.size(); ++i) {
        const int noise = static_cast<int>((i * 47 + 19) % 127) - 63;
        const double value = (i % 37) * 0.125 + ((i / 37) % 31) * 0.25 + noise * 0.03125;
        original[i] = static_cast<int32_t>(value);
    }
    auto input = original;
    size_t compressed_size = 0;
    std::unique_ptr<char[]> compressed(Falcons_compress(config, input.data(), compressed_size));
    Falcons::Config decoded_config;
    std::unique_ptr<int32_t[]> decoded(Falcons_decompress<int32_t>(decoded_config, compressed.get(), compressed_size));
    for (size_t i = 0; i < original.size(); ++i) {
        ASSERT_LE(std::abs(static_cast<double>(decoded[i]) - original[i]), config.absErrorBound)
            << "index=" << i << " original=" << original[i] << " recovered=" << decoded[i];
    }
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#else
    GTEST_SKIP() << "OpenMP is unavailable in this build";
#endif
}

TEST(V11WorkflowTest, DoesNotRejectDataVersionField) {
    auto input = make_values<float>(256);
    const auto original = input;
    Falcons::Config config(256);
    config.absErrorBound = 0.01;
    size_t compressed_size = 0;
    std::unique_ptr<char[]> compressed(Falcons_compress(config, input.data(), compressed_size));

    Falcons::Config header;
    const auto *read_position = reinterpret_cast<const unsigned char *>(compressed.get());
    size_t header_remaining = compressed_size;
    header.load(read_position, header_remaining);
    header.falconsDataVer = versionInt("9.9.9");
    auto *write_position = reinterpret_cast<unsigned char *>(compressed.get());
    header.save(write_position);

    Falcons::Config decoded_config;
    std::unique_ptr<float[]> decoded(Falcons_decompress<float>(decoded_config, compressed.get(), compressed_size));
    EXPECT_EQ(decoded_config.falconsDataVer, header.falconsDataVer);
    for (size_t i = 0; i < original.size(); ++i)
        ASSERT_LE(std::abs(static_cast<double>(decoded[i]) - original[i]), 0.01) << "index=" << i;
}
