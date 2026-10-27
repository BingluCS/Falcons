#include <cstdint>
#include <limits>
#include <type_traits>

#include "gtest/gtest.h"
#include "Falcons/utils/Interpolators.hpp"

TEST(InterpolatorTest, UsesWideIntegerIntermediates) {
    static_assert(std::is_same_v<Falcons::interpolation_wide_t<uint16_t>, int32_t>);
    static_assert(std::is_same_v<Falcons::interpolation_wide_t<int16_t>, int32_t>);
    static_assert(std::is_same_v<Falcons::interpolation_wide_t<uint32_t>, int64_t>);
    static_assert(std::is_same_v<Falcons::interpolation_wide_t<int32_t>, int64_t>);
    static_assert(std::is_same_v<Falcons::interpolation_wide_t<uint64_t>, __int128>);
    static_assert(std::is_same_v<Falcons::interpolation_wide_t<int64_t>, __int128>);

    constexpr uint64_t base = uint64_t{1} << 63;
    constexpr uint64_t a = base;
    constexpr uint64_t b = base + 20;
    constexpr uint64_t c = base + 40;
    constexpr uint64_t d = base + 60;

    EXPECT_EQ(Falcons::interp_linear(a, b), base + 10);
    EXPECT_EQ(Falcons::interp_linear1(a, b), base + 30);
    EXPECT_EQ(Falcons::interp_quad_1(a, b, c), base + 10);
    EXPECT_EQ(Falcons::interp_quad_2(a, b, c), base + 30);
    EXPECT_EQ(Falcons::interp_quad_3(a, b, c), base + 50);
    EXPECT_EQ(Falcons::interp_cubic(a, b, c, d), base + 30);
    EXPECT_EQ(Falcons::interp_cubic_natural(a, b, c, d), base + 30);
    EXPECT_EQ(Falcons::interp_cubic_front(a, b, c, d), base + 10);
    EXPECT_EQ(Falcons::interp_cubic_front_2(a, b, c, d), base + 5);
    EXPECT_EQ(Falcons::interp_cubic_back_1(a, b, c, d), base + 50);
    EXPECT_EQ(Falcons::interp_cubic_back_2(a, b, c, d), base + 70);
    EXPECT_EQ(Falcons::interp_cubic2(a, b, c, d), base + 30);
}

TEST(InterpolatorTest, SaturatesIntegralPredictions) {
    constexpr uint64_t umax = std::numeric_limits<uint64_t>::max();
    EXPECT_EQ(Falcons::interp_linear1<uint64_t>(100, 0), 0u);
    EXPECT_EQ(Falcons::interp_linear1<uint64_t>(0, umax), umax);
    EXPECT_EQ(Falcons::lorenzo_1d<uint64_t>(umax, 0), 0u);
    EXPECT_EQ(Falcons::lorenzo_1d<uint64_t>(0, umax), umax);

    constexpr int32_t imin = std::numeric_limits<int32_t>::min();
    constexpr int32_t imax = std::numeric_limits<int32_t>::max();
    EXPECT_EQ(Falcons::lorenzo_1d<int32_t>(imax, imin), imin);
    EXPECT_EQ(Falcons::lorenzo_1d<int32_t>(imin, imax), imax);
}

TEST(InterpolatorTest, AvoidsSignedInt32Overflow) {
    constexpr int32_t base = 2000000000;
    EXPECT_EQ(Falcons::interp_cubic<int32_t>(base, base + 20, base + 40, base + 60), base + 30);
    EXPECT_EQ(Falcons::interp_cubic_back_2<int32_t>(base, base + 20, base + 40, base + 60), base + 70);
}

TEST(InterpolatorTest, WidensSixteenBitInputsToInt32) {
    constexpr int16_t signed_base = 32000;
    EXPECT_EQ(Falcons::interp_cubic<int16_t>(signed_base, signed_base + 20,
                                         signed_base + 40, signed_base + 60),
              signed_base + 30);

    constexpr uint16_t unsigned_base = 65000;
    EXPECT_EQ(Falcons::interp_cubic<uint16_t>(unsigned_base, unsigned_base + 20,
                                          unsigned_base + 40, unsigned_base + 60),
              unsigned_base + 30);
}

TEST(InterpolatorTest, LeavesFloatingPredictionUnchanged) {
    EXPECT_FLOAT_EQ(Falcons::interp_linear(10.0f, 30.0f), 20.0f);
    EXPECT_FLOAT_EQ(Falcons::interp_cubic(0.0f, 20.0f, 40.0f, 60.0f), 30.0f);
}
