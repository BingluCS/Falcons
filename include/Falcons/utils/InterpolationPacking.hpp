#ifndef Falcons_INTERPOLATION_PACKING_HPP
#define Falcons_INTERPOLATION_PACKING_HPP

#include <cstddef>
#include <cstring>
#include <type_traits>
#ifdef __AVX2__
#include <immintrin.h>
#endif

// Packing-only switches; prediction, quantization and scheduling are unchanged.
#ifndef Falcons_PACKING_NATIVE
#define Falcons_PACKING_NATIVE 0
#endif
#ifndef Falcons_PACKING_YZ
#define Falcons_PACKING_YZ 1
#endif
#ifndef Falcons_PACKING_FLOAT_X
#define Falcons_PACKING_FLOAT_X 0
#endif

namespace Falcons::packing_detail {

template <class T>
inline bool use_yz(std::size_t stride, std::size_t count) noexcept {
#if defined(__AVX2__) && !Falcons_PACKING_NATIVE && Falcons_PACKING_YZ
    return (std::is_same_v<T, float> || std::is_same_v<T, double>) &&
           stride == 2 && count >= 32 / sizeof(T);
#else
    (void)stride;
    (void)count;
    return false;
#endif
}

template <class T>
inline bool use_float_x(std::size_t stride, std::size_t count) noexcept {
#if defined(__AVX2__) && !Falcons_PACKING_NATIVE && Falcons_PACKING_FLOAT_X
    return std::is_same_v<T, float> && stride == 1 && count >= 8;
#else
    (void)stride;
    (void)count;
    return false;
#endif
}

// Nonoverlapping buffers. Read through src[2*(count-1)], never the next element.
// Interleaved input elements must also be readable and not concurrently written.
template <class T>
inline void copy_stride2(T *dst, const T *src, std::size_t count) noexcept {
    std::size_t i = 0;
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
        for (; count - i >= 8; i += 8) {
            const auto a = _mm256_loadu_ps(src + 2 * i);
            const auto b = _mm256_loadu_ps(src + 2 * i + 7);
            // [0,2,8,10 | 4,6,12,14] -> [0,2,4,6,8,10,12,14].
            const auto pairs = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_ps(dst + i, _mm256_castpd_ps(
                _mm256_permute4x64_pd(_mm256_castps_pd(pairs), 0xd8)));
        }
    } else if constexpr (std::is_same_v<T, double>) {
        for (; count - i >= 4; i += 4) {
            const auto a = _mm256_loadu_pd(src + 2 * i);
            const auto b = _mm256_loadu_pd(src + 2 * i + 3);
            const auto pairs = _mm256_shuffle_pd(a, b, 0xa);
            _mm256_storeu_pd(dst + i, _mm256_permute4x64_pd(pairs, 0xd8));
        }
    }
#endif
    for (; i < count; ++i) std::memcpy(dst + i, src + 2 * i, sizeof(T));
}

} // namespace Falcons::packing_detail
#endif
