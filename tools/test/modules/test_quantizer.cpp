#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "gtest/gtest.h"
#include "Falcons/quantizer/LinearQuantizer.hpp"

template <typename Quantizer, typename T>
void runQuantizeRecoverTest() {
    T eb = 12.1973;
    Quantizer quantizer(eb);
    T data = static_cast<T>(100.11212);
    T data_ori = data;

    int quant_index = quantizer.quantize_and_overwrite(data, static_cast<T>(0));

    T recovered = quantizer.recover(static_cast<T>(0), quant_index);

    EXPECT_NEAR(recovered, data_ori, eb);
}

template <typename Quantizer, typename T>
void runFunctionalTest() {
    T eb = 12.1973;
    Quantizer quantizer(eb);

    const int N = 100;
    std::vector<T> originals(N);
    std::vector<int> quant_indices(N);

    // Generate 100 different input values.
    // We choose values that vary slightly so that quantization succeeds.
    for (int i = 0; i < N; i++) {
        T data = static_cast<T>(10.723 + (i * 0.0005));
        originals[i] = data;
        quant_indices[i] = quantizer.quantize_and_overwrite(data, static_cast<T>(0));
    }

    // Save the quantizer's state to a temporary buffer.
    std::vector<unsigned char> buffer(N * sizeof(T) * 4);
    unsigned char* save_ptr = buffer.data();
    quantizer.save(save_ptr);
    size_t saved_size = save_ptr - buffer.data();
    EXPECT_GT(saved_size, 0u) << "Saved state must be non-empty.";

    // Create a new quantizer instance and load the saved state.
    Quantizer loadedQuantizer(eb);  // The error bound will be overwritten by load().
    const unsigned char* load_ptr = buffer.data();
    size_t remaining_size = saved_size;
    loadedQuantizer.load(load_ptr, remaining_size);

    // Now, recover each value using the stored quantization indices.
    // Verify that the recovered value is within the error bound of the original.
    for (int i = 0; i < N; i++) {
        T recovered = loadedQuantizer.recover(static_cast<T>(0), quant_indices[i]);
        EXPECT_NEAR(recovered, originals[i], eb) << "Mismatch at index " << i;
    }
}

template <typename Quantizer, typename T>
void runAllTest() {
    runQuantizeRecoverTest<Quantizer, T>();
    runFunctionalTest<Quantizer, T>();
}

TEST(QuantizerTest, LinearQuantizer) {
    runAllTest<Falcons::LinearQuantizer<float>, float>();
}

TEST(QuantizerTest, Uint64ResidualsAboveDoublePrecision) {
    constexpr uint64_t base = (uint64_t{1} << 60) + 12345;
    Falcons::LinearQuantizer<uint64_t> quantizer(100.0);

    uint64_t below = base - 1000;
    int below_index = quantizer.quantize_and_overwrite(below, base);
    EXPECT_EQ(below_index, -5);
    EXPECT_EQ(below, base - 1000);
    EXPECT_EQ(quantizer.recover(base, below_index), base - 1000);

    uint64_t above = base + 1000;
    int above_index = quantizer.quantize_and_overwrite(above, base);
    EXPECT_EQ(above_index, 5);
    EXPECT_EQ(above, base + 1000);
    EXPECT_EQ(quantizer.recover(base, above_index), base + 1000);
}

TEST(QuantizerTest, Uint64NegativeReconstructionErrorIsPredictable) {
    constexpr uint64_t pred = (uint64_t{1} << 60) + 12345;
    constexpr uint64_t original = pred + 99;
    Falcons::LinearQuantizer<uint64_t> quantizer(100.0);

    uint64_t data = original;
    int quant_index = quantizer.quantize_and_overwrite(data, pred);
    EXPECT_EQ(quant_index, 0);
    EXPECT_EQ(data, pred);
    EXPECT_EQ(original - data, 99u);
    EXPECT_EQ(quantizer.recover(pred, quant_index), pred);
}

TEST(QuantizerTest, Uint64BoundariesAndLargeEscape) {
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    Falcons::LinearQuantizer<uint64_t> quantizer(100.0);

    uint64_t near_max = max - 1;
    int near_max_index = quantizer.quantize_and_overwrite(near_max, max - 1000);
    EXPECT_EQ(near_max_index, 5);
    EXPECT_EQ(near_max, max);
    EXPECT_EQ(quantizer.recover(max - 1000, near_max_index), max);

    uint64_t zero = 0;
    int escape = quantizer.quantize_and_overwrite(zero, max);
    EXPECT_EQ(escape, -32768);
    EXPECT_EQ(zero, 0u);
    EXPECT_EQ(quantizer.recover(max, escape), 0u);
}

TEST(QuantizerTest, Int64SignedResiduals) {
    constexpr int64_t base = (int64_t{1} << 60) + 12345;
    Falcons::LinearQuantizer<int64_t> quantizer(100.0);

    int64_t below = base - 1000;
    const int below_index = quantizer.quantize_and_overwrite(below, base);
    EXPECT_EQ(below_index, -5);
    EXPECT_LE(std::llabs(below - (base - 1000)), 100);
    EXPECT_EQ(quantizer.recover(base, below_index), below);

    int64_t above = base + 1000;
    const int above_index = quantizer.quantize_without_overwrite(above, base);
    EXPECT_EQ(above_index, 5);
    EXPECT_LE(std::llabs(quantizer.recover(base, above_index) - above), 100);

    int64_t extreme = std::numeric_limits<int64_t>::min();
    const int escape = quantizer.quantize_and_overwrite(extreme, std::numeric_limits<int64_t>::max());
    EXPECT_EQ(escape, -32768);
    EXPECT_EQ(quantizer.recover(std::numeric_limits<int64_t>::max(), escape),
              std::numeric_limits<int64_t>::min());

    int64_t unchanged = std::numeric_limits<int64_t>::max();
    const int zero_index = quantizer.quantize_and_overwrite(
        unchanged, std::numeric_limits<int64_t>::max());
    EXPECT_EQ(zero_index, 0);
    EXPECT_EQ(unchanged, std::numeric_limits<int64_t>::max());
}

TEST(QuantizerTest, IntegralStreamUidAndLegacyDecode) {
    Falcons::LinearQuantizer<uint64_t> quantizer(100.0);
    std::vector<unsigned char> buffer(256);
    unsigned char *save_ptr = buffer.data();
    quantizer.save(save_ptr);
    const size_t saved_size = static_cast<size_t>(save_ptr - buffer.data());
    EXPECT_EQ(buffer[0], 0b100);

    Falcons::LinearQuantizer<uint64_t> loaded(1.0);
    const unsigned char *load_ptr = buffer.data();
    size_t remaining = saved_size;
    loaded.load(load_ptr, remaining);
    constexpr uint64_t large_pred = (uint64_t{1} << 60) + 12345;
    constexpr int quant_index = 5;
    const uint64_t integral_reconstruction = large_pred + 1000;
    EXPECT_EQ(loaded.recover(large_pred, quant_index), integral_reconstruction);

    buffer[0] = 0b10;
    Falcons::LinearQuantizer<uint64_t> legacy(1.0);
    load_ptr = buffer.data();
    remaining = saved_size;
    legacy.load(load_ptr, remaining);
    const uint64_t legacy_reconstruction = static_cast<uint64_t>(
        std::fma(static_cast<double>(quant_index), 200.0, static_cast<double>(large_pred)));
    EXPECT_EQ(legacy.recover(large_pred, quant_index), legacy_reconstruction);
    EXPECT_NE(legacy_reconstruction, integral_reconstruction);

    Falcons::LinearQuantizer<float> float_quantizer(100.0);
    save_ptr = buffer.data();
    float_quantizer.save(save_ptr);
    EXPECT_EQ(buffer[0], 0b10);
}
