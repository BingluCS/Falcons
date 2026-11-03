#ifndef Falcons_BLOCKWISE_DECOMPOSITION_HPP
#define Falcons_BLOCKWISE_DECOMPOSITION_HPP

#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "Decomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/predictor/LorenzoPredictor.hpp"
#include "Falcons/predictor/Predictor.hpp"
#include "Falcons/quantizer/LinearQuantizer.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/FileUtil.hpp"
#include "Falcons/utils/BlockwiseIterator.hpp"
#include "Falcons/utils/Timer.hpp"

namespace Falcons {
template <class T, uint N, class Predictor, class Quantizer, class QuantIndex = int>
class BlockwiseDecomposition : public concepts::DecompositionInterface<T, QuantIndex, N> {
   public:
    using Block_iter = typename block_data<T, N>::block_iterator;

    BlockwiseDecomposition(const Config &conf, Predictor predictor, Quantizer quantizer)
        : predictor(predictor), quantizer(quantizer), fallback_predictor(conf.absErrorBound) {
        static_assert(std::is_base_of<concepts::PredictorInterface<T, N>, Predictor>::value,
                      "must implement the Predictor interface");
        static_assert(std::is_same<QuantIndex, int>::value || std::is_same<QuantIndex, int16_t>::value,
                      "unsupported blockwise quantization index type");
        validate_radius();
    }

    std::tuple<QuantIndex*, uint64_t> compress(const Config &conf, T *data) override {
        auto data_with_padding = std::make_shared<block_data<T, N>>(data, conf.dims, predictor.get_padding(), true);
        auto block = data_with_padding->block_iter(conf.blockSize);
        QuantIndex *quant_inds = new QuantIndex[conf.num];
        uint64_t cnt = 0;
        const int bias = std::is_same<QuantIndex, int16_t>::value ? 0 : quantizer.get_out_range().second / 2;
        do {
            concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
            if (!predictor.precompress(block)) {
                predictor_withfallback = &fallback_predictor;
            }
            predictor_withfallback->precompress_block_commit();
            Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                T pred = predictor_withfallback->predict(block, c, index);
                quant_inds[cnt++] = static_cast<QuantIndex>(quantizer.quantize_and_overwrite(*c, pred) + bias);
            });

        } while (block.next());
        return {quant_inds, cnt};
    }

    T *decompress(const Config &conf, QuantIndex *quant_inds, T *dec_data) override {
        validate_radius();
        QuantIndex *quant_inds_pos = quant_inds;
        const int bias = std::is_same<QuantIndex, int16_t>::value ? 0 : quantizer.get_out_range().second / 2;

        auto data_with_padding =
            std::make_shared<block_data<T, N>>(dec_data, conf.dims, predictor.get_padding(), false);
        auto block = data_with_padding->block_iter(conf.blockSize);
        do {
            concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
            if (!predictor.predecompress(block)) {
                predictor_withfallback = &fallback_predictor;
            }
            Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                T pred = predictor_withfallback->predict(block, c, index);
                *c = quantizer.recover(pred, static_cast<int>(*(quant_inds_pos++)) - bias);
            });

        } while (block.next());

        return dec_data;
    }

    void save(uchar *&c) override {
        fallback_predictor.save(c);
        predictor.save(c);
        quantizer.save(c);
    }

    void load(const uchar *&c, size_t &remaining_length) override {
        fallback_predictor.load(c, remaining_length);
        predictor.load(c, remaining_length);
        quantizer.load(c, remaining_length);
    }

    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

   private:
    void validate_radius() const {
        if constexpr (std::is_same<QuantIndex, int16_t>::value) {
            const int radius = quantizer.get_out_range().second / 2;
            if (radius < 1 || radius > static_cast<int>((std::numeric_limits<int16_t>::max)()) + 1)
                throw std::invalid_argument("blockwise quantization radius exceeds int16 range");
        }
    }

    Predictor predictor;
    Quantizer quantizer;
    LorenzoPredictor<T, N, 1> fallback_predictor;
};

template <class T, uint N, class QuantIndex = int, class Predictor, class Quantizer>
BlockwiseDecomposition<T, N, Predictor, Quantizer, QuantIndex> make_decomposition_blockwise(
    const Config &conf, Predictor predictor, Quantizer quantizer) {
    return BlockwiseDecomposition<T, N, Predictor, Quantizer, QuantIndex>(conf, predictor, quantizer);
}

}  // namespace Falcons
#endif
