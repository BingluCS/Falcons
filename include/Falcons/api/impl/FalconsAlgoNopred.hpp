#ifndef Falcons_FalconsALGO_NOPRED_HPP
#define Falcons_FalconsALGO_NOPRED_HPP

#include "Falcons/compressor/FalconsGenericCompressor.hpp"
#include "Falcons/decomposition/NoPredictionDecomposition.hpp"
#include "Falcons/encoder/HuffmanEncoder.hpp"
#include "Falcons/lossless/Lossless_zstd.hpp"
#include "Falcons/quantizer/LinearQuantizer.hpp"
#include "Falcons/utils/Config.hpp"

namespace Falcons {
template <class T, uint N>
size_t Falcons_compress_nopred(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
    assert(conf.cmprAlgo == ALGO_NOPRED);
    calAbsErrorBound(conf, data);

    auto falcons = make_compressor_falcons_generic<T, N>(
        make_decomposition_noprediction<T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
    return falcons->compress(conf, data, cmpData, cmpCap);
    //        return cmpData;
}

template <class T, uint N>
void Falcons_decompress_nopred(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == ALGO_NOPRED);
    auto cmpDataPos = cmpData;
    auto falcons = make_compressor_falcons_generic<T, N>(
        make_decomposition_noprediction<T, N>(conf, LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        HuffmanEncoder<int>(), Lossless_zstd());
    falcons->decompress(conf, cmpDataPos, cmpSize, decData);
}

}  // namespace Falcons
#endif
