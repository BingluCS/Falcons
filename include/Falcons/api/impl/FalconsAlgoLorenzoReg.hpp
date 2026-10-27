#ifndef Falcons_FalconsALGO_LORENZO_REG_HPP
#define Falcons_FalconsALGO_LORENZO_REG_HPP

#include <cmath>
#include <memory>

#include "Falcons/compressor/FalconsGenericCompressor.hpp"
#include "Falcons/decomposition/BlockwiseDecomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/lossless/Lossless_zstd.hpp"
#include "Falcons/predictor/ComposedPredictor.hpp"
#include "Falcons/predictor/LorenzoPredictor.hpp"
#include "Falcons/predictor/RegressionPredictor.hpp"
#include "Falcons/quantizer/LinearQuantizer.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/Iterator.hpp"
#include "Falcons/utils/QuantOptimizatioin.hpp"
#include "Falcons/utils/Statistic.hpp"

namespace Falcons {
template <class T, uint N, class Quantizer, class Encoder, class Lossless>
std::shared_ptr<concepts::CompressorInterface<T>> make_compressor_lorenzo_regression(const Config &conf,
                                                                                     Quantizer quantizer,
                                                                                     Encoder encoder,
                                                                                     Lossless lossless) {
    std::vector<std::shared_ptr<concepts::PredictorInterface<T, N>>> predictors;

    int methodCnt = (conf.lorenzo + conf.lorenzo2 + conf.regression);
    int use_single_predictor = (methodCnt == 1);
    if (methodCnt == 0) {
        throw std::invalid_argument("All lorenzo and regression methods are disabled.");
    }
    if (conf.lorenzo) {
        if (use_single_predictor) {
            return make_compressor_falcons_generic<T, N>(
                make_decomposition_blockwise<T, N>(conf, LorenzoPredictor<T, N, 1>(conf.absErrorBound), quantizer),
                encoder, lossless);
        } else {
            predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 1>>(conf.absErrorBound));
        }
    }
    if (conf.lorenzo2) {
        if (use_single_predictor) {
            return make_compressor_falcons_generic<T, N>(
                make_decomposition_blockwise<T, N>(conf, LorenzoPredictor<T, N, 2>(conf.absErrorBound), quantizer),
                encoder, lossless);
        } else {
            predictors.push_back(std::make_shared<LorenzoPredictor<T, N, 2>>(conf.absErrorBound));
        }
    }
    if (conf.regression) {
        if (use_single_predictor) {
            return make_compressor_falcons_generic<T, N>(
                make_decomposition_blockwise<T, N>(conf, RegressionPredictor<T, N>(conf.blockSize, conf.absErrorBound),
                                                   quantizer),
                encoder, lossless);
        } else {
            predictors.push_back(std::make_shared<RegressionPredictor<T, N>>(conf.blockSize, conf.absErrorBound));
        }
    }

    return make_compressor_falcons_generic<T, N>(
        make_decomposition_blockwise<T, N>(conf, ComposedPredictor<T, N>(predictors), quantizer), encoder, lossless);
}

template <class T, uint N>
size_t Falcons_compress_LorenzoReg(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
    assert(conf.cmprAlgo == ALGO_LORENZO_REG);
    calAbsErrorBound(conf, data);

    auto quantizer = LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2);
    auto falcons = make_compressor_lorenzo_regression<T, N>(conf, quantizer, HuffmanEncoder<int>(), Lossless_zstd());
    return falcons->compress(conf, data, cmpData, cmpCap);
}

template <class T, uint N>
void Falcons_decompress_LorenzoReg(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == ALGO_LORENZO_REG);
    auto cmpDataPos = cmpData;
    LinearQuantizer<T> quantizer;
    auto falcons = make_compressor_lorenzo_regression<T, N>(conf, quantizer, HuffmanEncoder<int>(), Lossless_zstd());
    falcons->decompress(conf, cmpDataPos, cmpSize, decData);
}
}  // namespace Falcons
#endif
