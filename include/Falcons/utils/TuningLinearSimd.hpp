#ifndef Falcons_TUNING_LINEAR_SIMD_HPP
#define Falcons_TUNING_LINEAR_SIMD_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef Falcons_TUNING_SIMD_LINEAR
#define Falcons_TUNING_SIMD_LINEAR 1
#endif
#ifndef Falcons_TUNING_SIMD_CUBIC
#define Falcons_TUNING_SIMD_CUBIC 1
#endif

#if (Falcons_TUNING_SIMD_LINEAR || Falcons_TUNING_SIMD_CUBIC) && defined(__AVX2__) && defined(__FMA__) && !defined(__FAST_MATH__)
#define Falcons_TUNING_SIMD_AVAILABLE 1
#include <immintrin.h>
#else
#define Falcons_TUNING_SIMD_AVAILABLE 0
#endif
#define Falcons_TUNING_LINEAR_AVAILABLE (Falcons_TUNING_SIMD_AVAILABLE && Falcons_TUNING_SIMD_LINEAR)
#define Falcons_TUNING_CUBIC_AVAILABLE (Falcons_TUNING_SIMD_AVAILABLE && Falcons_TUNING_SIMD_CUBIC)

namespace Falcons {
template <class T> class LinearQuantizer;

#if Falcons_TUNING_SIMD_AVAILABLE
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
namespace tuning_linear_detail {

struct Parameters {
    double bound, step, reciprocal;
    int radius;

    bool eligible() const {
        // Ignore sticky exception flags, but require masked traps, nearest, no FTZ/DAZ.
        return (_mm_getcsr() & 0xffc0u) == 0x1f80u &&
               bound > 0 && std::isnormal(bound) && step > 0 && std::isfinite(step) &&
               reciprocal > 0 && std::isfinite(reciprocal) && radius > 0 && radius <= 32768;
    }
};

template <class T> struct Result {
    T reconstructed[4];
    int codes[4];
    unsigned accepted;
};

inline bool finite(__m128 v) {
    const auto exponent = _mm_set1_epi32(0x7f800000);
    return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(
        _mm_and_si128(_mm_castps_si128(v), exponent), exponent))) == 0;
}

inline bool finite(__m256d v) {
    const auto exponent = _mm256_set1_epi64x(INT64_C(0x7ff0000000000000));
    return _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
        _mm256_and_si256(_mm256_castpd_si256(v), exponent), exponent))) == 0;
}

// Endpoint-safe loads: no padding or unselected endpoint is required.
inline __m128 load4(const float *p, std::size_t stride) {
    if (stride == 1) return _mm_loadu_ps(p);
    if (stride == 2)
        return _mm_shuffle_ps(_mm_loadu_ps(p), _mm_loadu_ps(p + 3), _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_setr_ps(p[0], p[stride], p[2 * stride], p[3 * stride]);
}

inline __m256d load4(const double *p, std::size_t stride) {
    if (stride == 1) return _mm256_loadu_pd(p);
    if (stride == 2) {
        const auto pairs = _mm256_shuffle_pd(_mm256_loadu_pd(p), _mm256_loadu_pd(p + 3), 0xa);
        return _mm256_permute4x64_pd(pairs, 0xd8);
    }
    return _mm256_setr_pd(p[0], p[stride], p[2 * stride], p[3 * stride]);
}

template <class T>
inline bool quantize4(__m256d original, __m256d prediction, __m256d difference,
                      const Parameters &p, Result<T> &out) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const auto rounded = _mm256_round_pd(_mm256_mul_pd(difference, _mm256_set1_pd(p.reciprocal)),
                                       _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
    const auto int_range = _mm256_and_pd(
        _mm256_cmp_pd(rounded, _mm256_set1_pd(-2147483648.0), _CMP_GE_OQ),
        _mm256_cmp_pd(rounded, _mm256_set1_pd(2147483647.0), _CMP_LE_OQ));
    // Defer out-of-range scalar float-to-int conversions to the unchanged caller.
    if (_mm256_movemask_pd(int_range) != 15) return false;
    const auto q = _mm256_cvttpd_epi32(rounded);
    const auto in_radius = _mm_and_si128(_mm_cmpgt_epi32(q, _mm_set1_epi32(-p.radius)),
                                       _mm_cmpgt_epi32(_mm_set1_epi32(p.radius), q));
    const auto mask = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(in_radius));
    // Convert through int, so rounded negative zero becomes positive integer zero.
    // Out-of-radius lanes do not run the scalar FMA; use benign operands there.
    const auto qd = _mm256_cvtepi32_pd(_mm_and_si128(q, in_radius));
    const auto pred = _mm256_blendv_pd(original, prediction, mask);
    const auto reconstructed = _mm256_fmadd_pd(qd, _mm256_set1_pd(p.step), pred);
    __m256d error;
    if constexpr (std::is_same_v<T, float>) {
        const auto narrowed = _mm256_cvtpd_ps(reconstructed);
        _mm_storeu_ps(out.reconstructed, narrowed);
        // Subtract in T, then promote the error; never round the double bound to float.
        error = _mm256_cvtps_pd(_mm_sub_ps(narrowed, _mm256_cvtpd_ps(original)));
    } else {
        _mm256_storeu_pd(out.reconstructed, reconstructed);
        error = _mm256_sub_pd(reconstructed, original);
    }
    const auto within_bound = _mm256_and_pd(
        _mm256_cmp_pd(error, _mm256_set1_pd(-p.bound), _CMP_GE_OQ),
        _mm256_cmp_pd(error, _mm256_set1_pd(p.bound), _CMP_LE_OQ));
    out.accepted = unsigned(_mm256_movemask_pd(_mm256_and_pd(mask, within_bound)));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out.codes), q);
    return true;
}

// Four independent targets in caller order. Return false without modifying data.
template <class T>
inline bool linear4(const T *data, std::size_t point_stride, std::size_t neighbor_stride,
                    const Parameters &p, Result<T> &out) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const auto left = load4(data - neighbor_stride, point_stride);
    const auto right = load4(data + neighbor_stride, point_stride);
    const auto original = load4(data, point_stride);
    if (!finite(left) || !finite(right) || !finite(original)) return false;
    if constexpr (std::is_same_v<T, float>) {
        const auto pred = _mm_mul_ps(_mm_add_ps(left, right), _mm_set1_ps(0.5f));
        const auto diff = _mm_sub_ps(original, pred);
        if (!finite(diff)) return false;
        return quantize4<T>(_mm256_cvtps_pd(original), _mm256_cvtps_pd(pred),
                            _mm256_cvtps_pd(diff), p, out);
    } else {
        const auto pred = _mm256_mul_pd(_mm256_add_pd(left, right), _mm256_set1_pd(0.5));
        const auto diff = _mm256_sub_pd(original, pred);
        if (!finite(diff)) return false;
        return quantize4<T>(original, pred, diff, p, out);
    }
}

template <class T>
inline bool cubic4(const T *data, std::size_t point_stride, std::size_t neighbor_stride,
                   const Parameters &p, Result<T> &out) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const auto a = load4(data - 3 * neighbor_stride, point_stride);
    const auto b = load4(data - neighbor_stride, point_stride);
    const auto c = load4(data + neighbor_stride, point_stride);
    const auto d = load4(data + 3 * neighbor_stride, point_stride);
    const auto original = load4(data, point_stride);
    if (!finite(a) || !finite(b) || !finite(c) || !finite(d) || !finite(original)) return false;
    // Match interp_cubic's T arithmetic and each rounding point, without FMA.
    if constexpr (std::is_same_v<T, float>) {
        auto prediction = _mm_mul_ps(_mm_add_ps(b, c), _mm_set1_ps(9.0f));
        prediction = _mm_sub_ps(prediction, a);
        prediction = _mm_sub_ps(prediction, d);
        prediction = _mm_mul_ps(prediction, _mm_set1_ps(0.0625f));
        const auto difference = _mm_sub_ps(original, prediction);
        if (!finite(difference)) return false;
        return quantize4<T>(_mm256_cvtps_pd(original), _mm256_cvtps_pd(prediction),
                            _mm256_cvtps_pd(difference), p, out);
    } else {
        auto prediction = _mm256_mul_pd(_mm256_add_pd(b, c), _mm256_set1_pd(9.0));
        prediction = _mm256_sub_pd(prediction, a);
        prediction = _mm256_sub_pd(prediction, d);
        prediction = _mm256_mul_pd(prediction, _mm256_set1_pd(0.0625));
        const auto difference = _mm256_sub_pd(original, prediction);
        if (!finite(difference)) return false;
        return quantize4<T>(original, prediction, difference, p, out);
    }
}

} // namespace tuning_linear_detail
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
#endif
} // namespace Falcons
#endif
