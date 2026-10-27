#include <cstdint>
#include <string>

#include "gtest/gtest.h"
#include "Falcons/utils/Statistic.hpp"

TEST(StatisticTest, ComputesUnsignedDifferencesWithoutUnderflow) {
    constexpr uint64_t base = uint64_t{1} << 63;
    EXPECT_DOUBLE_EQ(Falcons::statistic_difference(base + 99, base), 99.0);
    EXPECT_DOUBLE_EQ(Falcons::statistic_difference(base, base + 99), -99.0);
    EXPECT_DOUBLE_EQ(Falcons::statistic_difference(base + 499, base - 500), 999.0);
}

TEST(StatisticTest, VerifiesHighUint64RangeAndError) {
    constexpr uint64_t base = uint64_t{1} << 63;
    uint64_t original[] = {base - 500, base, base + 499};
    uint64_t reconstructed[] = {base - 499, base, base + 498};
    double psnr;
    double nrmse;
    double max_error;

    testing::internal::CaptureStdout();
    Falcons::verify(original, reconstructed, 3, psnr, nrmse, max_error);
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_DOUBLE_EQ(max_error, 1.0);
    EXPECT_NE(output.find("Min=9223372036854775308, Max=9223372036854776307, range=999"),
              std::string::npos);
}
