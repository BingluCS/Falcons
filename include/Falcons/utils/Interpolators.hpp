/**
 * @file Interpolators.hpp
 * @author Bing Lu
 * @brief Falcons scalar and SIMD-consistent interpolation helpers.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_INTERPOLATORS_HPP
#define Falcons_INTERPOLATORS_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "Falcons/def.hpp"

// Bit-exactness across scalar / AVX2 / SVE2 requires the prediction helpers to NOT be
// auto-fused into FMAs (which happens by default on ARM with -ffp-contract=fast but not
// on plain x86). Fused ops are expressed explicitly with std::fma; everything else must
// stay unfused. This does NOT affect the SIMD kernels (they use explicit intrinsics).
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace Falcons {

template <class T>
using interpolation_wide_t = std::conditional_t<
    (sizeof(T) <= sizeof(int16_t)), int32_t,
    std::conditional_t<(sizeof(T) <= sizeof(int32_t)), int64_t, __int128>>;

template <class T>
ALWAYS_INLINE T interp_narrow(interpolation_wide_t<T> value) {
    static_assert(std::is_integral_v<T>);
    using Wide = interpolation_wide_t<T>;
    const Wide lower = static_cast<Wide>(std::numeric_limits<T>::lowest());
    const Wide upper = static_cast<Wide>(std::numeric_limits<T>::max());
    if (value < lower)
        return std::numeric_limits<T>::lowest();
    if (value > upper)
        return std::numeric_limits<T>::max();
    return static_cast<T>(value);
}

template <class T>
ALWAYS_INLINE T interp_linear(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((static_cast<Wide>(a) + static_cast<Wide>(b)) / 2);
    }
    return (a + b) / 2;
}

// Floating branches below are written to match the SIMD (AVX2/SVE2) inline op
// sequence exactly; integral branches evaluate the same polynomials without overflow.
template <class T>
ALWAYS_INLINE T interp_linear1(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((-static_cast<Wide>(a) + 3 * static_cast<Wide>(b)) / 2);
    }
    // 1.5*b - 0.5*a   (SVE: svmls_n(1.5*b, a, 0.5))
    return std::fma(-T(0.5), a, T(1.5) * b);
}

template <class T>
ALWAYS_INLINE T interp_quad_1(T a, T b, T c) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((3 * static_cast<Wide>(a) + 6 * static_cast<Wide>(b) -
                                 static_cast<Wide>(c)) / 8);
    }
    // (3a + 6b - c) * 0.125   (SVE: svmla_n(svnmls_n(c,b,6), a, 3) * 0.125)
    T t = std::fma(T(6), b, -c);   // 6*b - c
    t = std::fma(T(3), a, t);      // 3*a + (6*b - c)
    return t * T(0.125);
}

template <class T>
ALWAYS_INLINE T interp_quad_2(T a, T b, T c) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((-static_cast<Wide>(a) + 6 * static_cast<Wide>(b) +
                                 3 * static_cast<Wide>(c)) / 8);
    }
    // (-a + 6b + 3c) * 0.125   (SVE: svmla_n(svnmls_n(a,b,6), c, 3) * 0.125)
    T t = std::fma(T(6), b, -a);   // 6*b - a
    t = std::fma(T(3), c, t);      // 3*c + (6*b - a)
    return t * T(0.125);
}

template <class T>
ALWAYS_INLINE T interp_quad_3(T a, T b, T c) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((3 * static_cast<Wide>(a) - 10 * static_cast<Wide>(b) +
                                 15 * static_cast<Wide>(c)) / 8);
    }
    return (3 * a - 10 * b + 15 * c) / 8;
}

template <class T>
ALWAYS_INLINE T interp_cubic(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((-static_cast<Wide>(a) + 9 * static_cast<Wide>(b) +
                                 9 * static_cast<Wide>(c) - static_cast<Wide>(d)) / 16);
    }
    // ((b+c)*9 - a - d) * 0.0625   (SVE/AVX2 inline order; no FMA)
    T t = (b + c) * T(9);
    t = t - a;
    t = t - d;
    return t * T(0.0625);
}

template <class T>
ALWAYS_INLINE T interp_cubic_natural(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((23 * (static_cast<Wide>(b) + static_cast<Wide>(c)) -
                                 3 * (static_cast<Wide>(a) + static_cast<Wide>(d))) / 40);
    }
   return 0.575 * (b + c) - 0.075 * (a + d);
}

template<class T>
ALWAYS_INLINE T lorenzo_1d(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>(2 * static_cast<Wide>(b) - static_cast<Wide>(a));
    }
    return 2 * b - a;
}

template<class T>
ALWAYS_INLINE T lorenzo_2d(T a, T b, T c) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>(static_cast<Wide>(b) + static_cast<Wide>(c) - static_cast<Wide>(a));
    }
    return (b + c - a);
}

template<class T>
ALWAYS_INLINE T lorenzo_3d(T a, T b, T c, T d, T e,T f,T g) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>(static_cast<Wide>(a) - static_cast<Wide>(b) - static_cast<Wide>(c) +
                                static_cast<Wide>(d) - static_cast<Wide>(e) + static_cast<Wide>(f) +
                                static_cast<Wide>(g));
    }
    return (a - b - c + d - e + f + g);
}

    


template <class T>
ALWAYS_INLINE T interp_cubic_front(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((5 * static_cast<Wide>(a) + 15 * static_cast<Wide>(b) -
                                 5 * static_cast<Wide>(c) + static_cast<Wide>(d)) / 16);
    }
    return (5 * a + 15 * b - 5 * c + d) / 16;
}

template <class T>
ALWAYS_INLINE T interp_cubic_front_2(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((static_cast<Wide>(a) + 6 * static_cast<Wide>(b) -
                                 4 * static_cast<Wide>(c) + static_cast<Wide>(d)) / 4);
    }
    return (a + 6 * b - 4 * c + d) / 4;
}

template <class T>
ALWAYS_INLINE T interp_cubic_back_1(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((static_cast<Wide>(a) - 5 * static_cast<Wide>(b) +
                                 15 * static_cast<Wide>(c) + 5 * static_cast<Wide>(d)) / 16);
    }
    return (a - 5 * b + 15 * c + 5 * d) / 16;
}

template <class T>
ALWAYS_INLINE T interp_cubic_back_2(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((-5 * static_cast<Wide>(a) + 21 * static_cast<Wide>(b) -
                                 35 * static_cast<Wide>(c) + 35 * static_cast<Wide>(d)) / 16);
    }
    return (-5 * a + 21 * b - 35 * c + 35 * d) / 16;
}

template <class T>
ALWAYS_INLINE T interp_cubic2(T a, T b, T c, T d) {
    if constexpr (std::is_integral_v<T>) {
        using Wide = interpolation_wide_t<T>;
        return interp_narrow<T>((-3 * static_cast<Wide>(a) + 23 * static_cast<Wide>(b) +
                                 23 * static_cast<Wide>(c) - 3 * static_cast<Wide>(d)) / 40);
    }
    return (-3 * a + 23 * b + 23 * c - 3 * d) / 40;
}

template <class T>
ALWAYS_INLINE T interp_akima(T a, T b, T c, T d) {
    T t0 = 2 * b - a - c;
    T t1 = 2 * c - b - d;
    T abt0 = fabs(t0);
    T abt1 = fabs(t1);
    if (fabs(abt0 + abt1) > 1e-9) {
        return (b + c) / 2 + (t0 * abt1 + t1 * abt0) / 8 / (abt0 + abt1);
    } else {
        return (b + c) / 2;
    }
}

template <class T>
ALWAYS_INLINE T interp_pchip(T a, T b, T c, T d) {
    T pchip = (b + c) / 2;
    if ((b - a < 0) == (c - b < 0) && fabs(c - a) > 1e-9) {
        pchip += 1 / 4 * (b - a) * (c - b) / (c - a);
    }
    if ((c - b < 0) == (d - c < 0) && fabs(d - b) > 1e-9) {
        pchip -= 1 / 4 * (c - b) * (d - c) / (d - b);
    }
    return pchip;
}

}  // namespace Falcons
#endif  // Falcons_INTERPOLATORS_HPP
