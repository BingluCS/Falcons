#ifndef Falcons_IMPL_FalconsDISPATCHER_HPP
#define Falcons_IMPL_FalconsDISPATCHER_HPP

#include "Falcons/api/impl/FalconsAlgoInterp.hpp"
#include "Falcons/api/impl/FalconsAlgoLorenzoReg.hpp"
#include "Falcons/api/impl/FalconsAlgoNopred.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/Statistic.hpp"
#include "Falcons/utils/Timer.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Falcons {
template <class T, uint N>
size_t Falcons_compress_dispatcher(Config &conf, const T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
#ifdef Falcons_PRINT_TIMINGS
            Timer timer(true);
#endif

calAbsErrorBound(conf, data);
#ifdef Falcons_PRINT_TIMINGS
            timer.stop("cal rel ");
#endif
    size_t cmpSize = 0;

    // if absErrorBound is 0, use lossless only mode
    if (conf.absErrorBound == 0) {
        conf.cmprAlgo = ALGO_LOSSLESS;
    }

    // do lossy compression
    bool isCmpCapSufficient = true;
    if (conf.cmprAlgo != ALGO_LOSSLESS) {
        // size_t alignment = 256; 
        // T * dataCopy = nullptr;

        try {
// #ifdef Falcons_PRINT_TIMINGS
//             Timer timer(true);
// #endif
//            // std::vector<T> dataCopy(data, data + conf.num);
//             auto n = conf.num;
//             // dataCopy = new T[n];
//             dataCopy= static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignment)));

//             #ifdef _OPENMP

//                 int max_threads = omp_get_max_threads();
//                 auto n_threads = std::min(max_threads, static_cast<int>(n) / 65536);

//                 if(n_threads > 1){
//                     omp_set_num_threads(n_threads);
//                     #pragma omp parallel
//                     {
//                         int tid  = omp_get_thread_num();
//                         int nth  = omp_get_num_threads();

//                         std::size_t begin =  static_cast<std::size_t>(tid) * n /  static_cast<std::size_t>(nth);
//                         std::size_t end   =  static_cast<std::size_t>(tid + 1) * n / static_cast<std::size_t>(nth);

//                         std::size_t len   = end - begin;
//                         if (len > 0) {
//                             std::memcpy(dataCopy + begin, data + begin, len * sizeof(T));
//                         }
//                     }
//                     omp_set_num_threads(max_threads);
//                 }
//                 else
//                     std::memcpy(dataCopy, data, n * sizeof(T));
//             #else
//                 std::memcpy(dataCopy, data, n * sizeof(T));
//             #endif
// #ifdef Falcons_PRINT_TIMINGS
//              timer.stop("datacopy");
// #endif
#ifdef Falcons_PRINT_TIMINGS
             timer.start();
#endif
            if (conf.cmprAlgo == ALGO_LORENZO_REG) {
                cmpSize = Falcons_compress_LorenzoReg<T, N>(conf, const_cast<T *>(data), cmpData, cmpCap);
            } else if (conf.cmprAlgo == ALGO_INTERP) {
                cmpSize = Falcons_compress_Interp<T, N>(conf, const_cast<T *>(data), cmpData, cmpCap);
            } else if (conf.cmprAlgo == ALGO_INTERP_LORENZO) {
                cmpSize = Falcons_compress_Interp_lorenzo<T, N>(conf, const_cast<T *>(data), cmpData, cmpCap);
            } else if (conf.cmprAlgo == ALGO_NOPRED) {
                cmpSize = Falcons_compress_nopred<T, N>(conf, const_cast<T *>(data), cmpData, cmpCap);
            } else {
                throw std::invalid_argument("Unknown compression algorithm");
            }
#ifdef Falcons_PRINT_TIMINGS
             timer.stop("total cmp");
            //  timer.start();
#endif
            // ::operator delete(dataCopy, std::align_val_t(alignment));
// #ifdef Falcons_PRINT_TIMINGS
//              timer.stop("delete");
// #endif
        } catch (std::length_error &e) {
            // if(dataCopy)
            //     ::operator delete(dataCopy, std::align_val_t(alignment));
            if (std::string(e.what()) == Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH) {
                isCmpCapSufficient = false;
                printf("Falcons is downgraded to lossless mode because the buffer for compressed data is not large enough.\n");
            } else {
                throw;
            }
        }

        
    }
    // do lossless only compression if 1) cmpr algorithm is lossless or 2) compressed buffer not large enough for lossy
    if (conf.cmprAlgo == ALGO_LOSSLESS || !isCmpCapSufficient) {
        conf.cmprAlgo = ALGO_LOSSLESS;
        auto zstd = Lossless_zstd();
        return zstd.compress(reinterpret_cast<const uchar *>(data), conf.num * sizeof(T), cmpData, cmpCap);
    }

    // if lossy compression ratio < 3, test if lossless only mode has a better ratio than lossy
    if (conf.num * sizeof(T) / 1.0 / cmpSize < 3) {
        auto zstd = Lossless_zstd();
        auto zstdCmpCap = ZSTD_compressBound(conf.num * sizeof(T)) + sizeof(size_t);
        auto zstdCmpData = static_cast<uchar *>(malloc(zstdCmpCap));
        size_t zstdCmpSize =
            zstd.compress(reinterpret_cast<const uchar *>(data), conf.num * sizeof(T), zstdCmpData, zstdCmpCap);
        if (zstdCmpSize < cmpSize && zstdCmpSize <= cmpCap) {
            conf.cmprAlgo = ALGO_LOSSLESS;
            memcpy(cmpData, zstdCmpData, zstdCmpSize);
            cmpSize = zstdCmpSize;
        }
        free(zstdCmpData);
    }
    return cmpSize;
}

template <class T, uint N>
void Falcons_decompress_dispatcher(Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    if (conf.cmprAlgo == ALGO_LOSSLESS) {
        auto zstd = Lossless_zstd();
        size_t decDataSize = 0;
        auto decDataPos = reinterpret_cast<uchar *>(decData);
        zstd.decompress(cmpData, cmpSize, decDataPos, decDataSize);
        if (decDataSize != conf.num * sizeof(T)) {
            throw std::runtime_error("Decompressed data size does not match the original data size");
        }
    } else if (conf.cmprAlgo == ALGO_LORENZO_REG) {
        Falcons_decompress_LorenzoReg<T, N>(conf, cmpData, cmpSize, decData);
    } else if (conf.cmprAlgo == ALGO_INTERP) {
        Falcons_decompress_Interp<T, N>(conf, cmpData, cmpSize, decData);
    } else if (conf.cmprAlgo == ALGO_NOPRED) {
        Falcons_decompress_nopred<T, N>(conf, cmpData, cmpSize, decData);
    } else {
        throw std::invalid_argument("Unknown compression algorithm");
    }
}
}  // namespace Falcons
#endif
