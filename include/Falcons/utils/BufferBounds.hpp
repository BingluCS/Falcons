#ifndef FALCONS_BUFFER_BOUNDS_HPP
#define FALCONS_BUFFER_BOUNDS_HPP

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <zstd.h>

namespace Falcons::buffer_bounds {

inline size_t add(size_t a, size_t b) {
    const size_t limit = static_cast<size_t>((std::numeric_limits<ptrdiff_t>::max)());
    if (a > limit || b > limit - a) throw std::length_error("compression capacity overflow");
    return a + b;
}

inline size_t mul(size_t a, size_t b) {
    const size_t limit = static_cast<size_t>((std::numeric_limits<ptrdiff_t>::max)());
    if (a && b > limit / a) throw std::length_error("compression capacity overflow");
    return a * b;
}

template<class T>
size_t scratch_bound(size_t data_count, size_t quant_count) {
    return add(add(mul(data_count, sizeof(T)), 4096),
               add(mul(quant_count, 8), size_t(4) << 20));
}

template<class T>
size_t compressed_bound(size_t count, size_t config_bytes) {
    const size_t frame = scratch_bound<T>(count, count);
    const size_t bound = ZSTD_compressBound(frame);
    if (ZSTD_isError(bound)) throw std::length_error("Zstd capacity overflow");
    return add(add(config_bytes, sizeof(size_t)), bound);
}

}  // namespace Falcons::buffer_bounds
#endif
