#ifndef Falcons_NO_PREDICTION_DECOMPOSITION_HPP
#define Falcons_NO_PREDICTION_DECOMPOSITION_HPP

#include "Decomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/quantizer/Quantizer.hpp"
#include "Falcons/utils/Config.hpp"

namespace Falcons {
template <class T, uint N, class Quantizer>
class NoPredictionDecomposition : public concepts::DecompositionInterface<T, int, N> {
   public:
    NoPredictionDecomposition(const Config &conf, Quantizer quantizer) : quantizer(quantizer) {
        static_assert(std::is_base_of<concepts::QuantizerInterface<T, int>, Quantizer>::value,
                      "must implement the quantizer interface");
    }

    T *decompress(const Config &conf, int* quant_inds, T *dec_data) override {
        for (size_t i = 0; i < conf.num; i++) {
            dec_data[i] = quantizer.recover(0, quant_inds[i]);
        }
        quantizer.postdecompress_data();
        return dec_data;
    }

    std::tuple<int*, uint64_t> compress(const Config &conf, T *data) override {
        std::vector<int> quant_inds(conf.num);
        for (size_t i = 0; i < conf.num; i++) {
            quant_inds[i] = quantizer.quantize_and_overwrite(data[i], 0);
        }
        quantizer.postcompress_data();
        return {quant_inds.data(), static_cast<uint64_t>(quant_inds.size())};
    }

    void save(uchar *&c) override { quantizer.save(c); }

    void load(const uchar *&c, size_t &remaining_length) override { quantizer.load(c, remaining_length); }

    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

   private:
    Quantizer quantizer;
};

template <class T, uint N, class Quantizer>
NoPredictionDecomposition<T, N, Quantizer> make_decomposition_noprediction(const Config &conf, Quantizer quantizer) {
    return NoPredictionDecomposition<T, N, Quantizer>(conf, quantizer);
}

}  // namespace Falcons

#endif
