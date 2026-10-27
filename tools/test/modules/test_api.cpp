#include <cmath>
#include <memory>
#include <vector>
#include "Falcons/api/falcons.hpp"
#include "gtest/gtest.h"

TEST(ApiTest, SmallArrayRoundTripUsesBound) {
    std::vector<float> values(256);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(i * 0.01f);
    auto original = values;
    Falcons::Config config(256);
    config.absErrorBound = 1e-3;
    std::vector<char> bytes(Falcons::Falcons_compress_size_bound<float>(config));
    const auto size = Falcons_compress(config, values.data(), bytes.data(), bytes.size());
    std::unique_ptr<float[]> decoded(Falcons_decompress<float>(config, bytes.data(), size));
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_LE(std::abs(original[i] - decoded[i]), 1e-3);
}

TEST(ApiTest, RejectsTruncatedHeaders) {
    Falcons::Config config(256);
    std::vector<unsigned char> header(config.size_est());
    auto position = header.data();
    config.save(position);
    for (size_t n = 0; n <= header.size(); ++n) {
        std::vector<char> truncated(header.begin(), header.begin() + n);
        EXPECT_THROW(Falcons_read_config(config, truncated.data(), truncated.size()), std::invalid_argument);
    }
}

TEST(ApiTest, RejectsSmallOutputBuffer) {
    std::vector<float> values(256, 1.5f);
    Falcons::Config config(256);
    size_t size = 0;
    std::unique_ptr<char[]> bytes(Falcons_compress(config, values.data(), size));
    float guard = 42.0f;
    float *output = &guard;
    EXPECT_THROW(Falcons_decompress(config, bytes.get(), size, output, 1), std::invalid_argument);
    EXPECT_EQ(guard, 42.0f);
}

TEST(ApiTest, RejectsTruncatedPayload) {
    std::vector<float> values(256, 1.5f);
    Falcons::Config config(256);
    size_t size = 0;
    std::unique_ptr<char[]> bytes(Falcons_compress(config, values.data(), size));
    const size_t header = static_cast<unsigned char>(bytes[0]);
    for (size_t n = header + 1; n < size; ++n) {
        std::vector<char> truncated(bytes.get(), bytes.get() + n);
        std::vector<float> output(256);
        float *pointer = output.data();
        EXPECT_THROW(Falcons_decompress(config, truncated.data(), n, pointer, output.size()), std::invalid_argument);
    }
}
