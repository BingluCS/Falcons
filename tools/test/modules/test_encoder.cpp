#include <cmath>
#include <cstdint>
#include <memory>

#include "Falcons/encoder/ArithmeticEncoder.hpp"
#include "Falcons/encoder/BypassEncoder.hpp"
#include "Falcons/encoder/HuffmanEncoder.hpp"
#include "Falcons/encoder/RunlengthEncoder.hpp"
#include "Falcons/encoder/RleFseEncoder.hpp"
#include "gtest/gtest.h"

template <typename Encoder, typename T>
void runFunctionalTest() {
    int N = 1000;
    std::vector<Falcons::uchar> buffer_data(N * sizeof(T) * 2);
    std::vector<Falcons::uchar> buffer_conf(N * sizeof(T) * 2);
    std::vector<T> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i % 100;
    }

    size_t data_len = 0, conf_len = 0;
    {
        Falcons::uchar *buffer_data_pos = buffer_data.data();
        Falcons::uchar *buffer_conf_pos = buffer_conf.data();
        Encoder coder;
        coder.preprocess_encode(data, 100);
        coder.encode(data, buffer_data_pos);
        coder.save(buffer_conf_pos);
        data_len = buffer_data_pos - buffer_data.data();
        conf_len = buffer_conf_pos - buffer_conf.data();
    }
    {
        const Falcons::uchar *buffer_data_pos = buffer_data.data();
        const Falcons::uchar *buffer_conf_pos = buffer_conf.data();
        Encoder coder;
        coder.load(buffer_conf_pos, conf_len);
        std::unique_ptr<T[]> dataDecoded(coder.decode(buffer_data_pos, N));
        for (int i = 0; i < N; i++) {
            EXPECT_EQ(data[i], dataDecoded[i]);
        }
    }
}

template <typename Encoder, typename T>
void runAllTest() {
    runFunctionalTest<Encoder, T>();
}

TEST(EncoderTest, HuffmanEncoder) { runAllTest<Falcons::HuffmanEncoder<int>, int>(); }

TEST(EncoderTest, RunlengthEncoder) { runAllTest<Falcons::RunlengthEncoder<int>, int>(); }

TEST(EncoderTest, ArithmeticEncoder) { runAllTest<Falcons::ArithmeticEncoder<int>, int>(); }

TEST(EncoderTest, BypassEncoder) { runAllTest<Falcons::BypassEncoder<int>, int>(); }

TEST(EncoderTest, RleFseEncoder) { runAllTest<Falcons::RleFseEncoder<int16_t>, int16_t>(); }
