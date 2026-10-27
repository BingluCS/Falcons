/**
 * @file FalconsImpl.hpp
 * @author Bing Lu
 * @brief Falcons compression and decompression dispatch.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_IMPL_Falcons_HPP
#define Falcons_IMPL_Falcons_HPP

#include "Falcons/api/impl/FalconsDispatcher.hpp"
#include "Falcons/api/impl/FalconsImplOMP.hpp"
#include "Falcons/def.hpp"

namespace Falcons {
template <class T, uint N>
size_t Falcons_compress_impl(Config &conf, const T *data, uchar *cmpData, size_t cmpCap) {
    return Falcons_compress_dispatcher<T, N>(conf, data, cmpData, cmpCap);
// #ifndef _OPENMP
//     conf.openmp = false;
// #endif
//     if (conf.openmp) {
//         return Falcons_compress_OMP<T, N>(conf, data, cmpData, cmpCap);
//     } else {
//         return Falcons_compress_dispatcher<T, N>(conf, data, cmpData, cmpCap);
//     }
}

template <class T, uint N>
void Falcons_decompress_impl(Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    Falcons_decompress_dispatcher<T, N>(conf, cmpData, cmpSize, decData);
// #ifndef _OPENMP
//     conf.openmp = false;
// #endif
//     if (conf.openmp) {
//         Falcons_decompress_OMP<T, N>(conf, cmpData, cmpSize, decData);
//     } else {
//         Falcons_decompress_dispatcher<T, N>(conf, cmpData, cmpSize, decData);
//     }
}

template <class T>
size_t Falcons_compress_size_bound(const Config &conf) {
    bool omp = conf.openmp;
#ifndef _OPENMP
    omp = false;
#endif
    if (omp) {
        return 4096 + Falcons_compress_size_bound_omp<T>(conf);
    } else {
        return 4096 + conf.size_est() + ZSTD_compressBound(conf.num * sizeof(T));
    }
}

}  // namespace Falcons
#endif
