#include "Falcons/api/impl/FalconsAlgoLorenzoReg.hpp"
#include "Falcons/encoder/HuffmanEncoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static void run_case(int kind, int quant_bins = 65536) {
    const size_t n = (size_t(1) << 20) + 37;
    Falcons::Config conf(n);
    conf.cmprAlgo = Falcons::ALGO_LORENZO_REG;
    conf.errorBoundMode = Falcons::EB_ABS;
    conf.absErrorBound = 0.01;
    conf.blockSize = 16;
    conf.quantbinCnt = quant_bins;
    std::vector<float> input(n);
    uint32_t rng = 7;
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        input[i] = kind == 0 ? 0.0f : kind == 1
            ? static_cast<float>(static_cast<int>(rng >> 16) - 32768) * 0.003f
            : (i & 1 ? 1000.0f : -1000.0f);
    }
    auto old_data = input, new_data = input;
    auto old_dec = Falcons::make_decomposition_blockwise<float, 1>(
        conf, Falcons::LorenzoPredictor<float, 1, 1>(conf.absErrorBound),
        Falcons::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2));
    auto new_dec = Falcons::make_decomposition_blockwise<float, 1, int16_t>(
        conf, Falcons::LorenzoPredictor<float, 1, 1>(conf.absErrorBound),
        Falcons::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2));
    auto [old_codes_raw, old_n] = old_dec.compress(conf, old_data.data());
    auto [new_codes_raw, new_n] = new_dec.compress(conf, new_data.data());
    std::unique_ptr<int[]> old_codes(old_codes_raw);
    std::unique_ptr<int16_t[]> new_codes(new_codes_raw);
    check(old_n == n && new_n == n && old_data == new_data, "prediction differs");
    size_t unpredictable = 0;
    for (size_t i = 0; i < n; ++i) {
        check(old_codes[i] == int(new_codes[i]) + quant_bins / 2, "quantization bias differs");
        unpredictable += new_codes[i] == -quant_bins / 2;
    }
    if (kind == 2) check(unpredictable > n / 2, "unpredictable values not exercised");
    Falcons::RleFseEncoder<int16_t> probe;
    probe.preprocess_encode(new_codes.get(), n, conf.quantbinCnt);
    Falcons::uchar header[5], *header_pos = header;
    probe.save(header_pos);
    const unsigned mode = header[0] & 0x7f;
    check((kind == 0 && mode == 0) || (kind == 1 && mode == 2) ||
          (kind == 2 && (mode == 1 || mode == 2)), "unexpected lossless route");

    auto reference = Falcons::make_compressor_falcons_generic<float, 1>(
        Falcons::make_decomposition_blockwise<float, 1>(
            conf, Falcons::LorenzoPredictor<float, 1, 1>(conf.absErrorBound),
            Falcons::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2)),
        Falcons::HuffmanEncoder<int>(), Falcons::Lossless_zstd());
    auto candidate = Falcons::make_compressor_falcons_generic<float, 1>(
        Falcons::make_decomposition_blockwise<float, 1, int16_t>(
            conf, Falcons::LorenzoPredictor<float, 1, 1>(conf.absErrorBound),
            Falcons::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2)),
        Falcons::RleFseEncoder<int16_t>(), Falcons::Lossless_zstd());
    const size_t capacity = Falcons::buffer_bounds::compressed_bound<float>(n, conf.size_est());
    std::vector<Falcons::uchar> old_bytes(capacity), new_bytes(capacity);
    old_data = input; new_data = input;
    const size_t old_size = reference->compress(conf, old_data.data(), old_bytes.data(), capacity);
    const size_t new_size = candidate->compress(conf, new_data.data(), new_bytes.data(), capacity);
    std::vector<float> old_out(n), new_out(n);
    reference->decompress(conf, old_bytes.data(), old_size, old_out.data());
    candidate->decompress(conf, new_bytes.data(), new_size, new_out.data());
    check(old_out == new_out, "reconstruction differs from Huffman reference");
    for (size_t i = 0; i < n; ++i)
        check(std::isfinite(new_out[i]) && std::abs(double(input[i]) - new_out[i]) <= conf.absErrorBound,
              "error bound exceeded");
    std::printf("PASS kind=%d bins=%d mode=%u unpredictable=%zu huffman=%zu rlefse=%zu\n",
                kind, quant_bins, unsigned(header[0] & 0x7f), unpredictable, old_size, new_size);
}

static void test_composed_predictor() {
    const size_t n = (size_t(1) << 17) + 3;
    Falcons::Config conf(n);
    conf.cmprAlgo = Falcons::ALGO_LORENZO_REG;
    conf.errorBoundMode = Falcons::EB_ABS;
    conf.absErrorBound = 0.01;
    conf.blockSize = 5;
    conf.lorenzo = true;
    conf.lorenzo2 = true;
    conf.regression = false;
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i)
        input[i] = static_cast<float>(20.0 * std::sin(i * 0.113) + double(i % 97));
    std::vector<std::shared_ptr<Falcons::concepts::PredictorInterface<float, 1>>> predictors;
    predictors.push_back(std::make_shared<Falcons::LorenzoPredictor<float, 1, 1>>(conf.absErrorBound));
    predictors.push_back(std::make_shared<Falcons::LorenzoPredictor<float, 1, 2>>(conf.absErrorBound));
    auto reference = Falcons::make_compressor_falcons_generic<float, 1>(
        Falcons::make_decomposition_blockwise<float, 1>(conf, Falcons::ComposedPredictor<float, 1>(predictors),
            Falcons::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2)),
        Falcons::HuffmanEncoder<int>(), Falcons::Lossless_zstd());
    const size_t capacity = Falcons::buffer_bounds::compressed_bound<float>(n, conf.size_est());
    std::vector<Falcons::uchar> old_bytes(capacity), new_bytes(capacity);
    auto old_data = input, new_data = input;
    const size_t old_size = reference->compress(conf, old_data.data(), old_bytes.data(), capacity);
    const size_t new_size = Falcons::Falcons_compress_LorenzoReg<float, 1>(conf, new_data.data(), new_bytes.data(), capacity);
    std::vector<float> old_out(n), new_out(n);
    reference->decompress(conf, old_bytes.data(), old_size, old_out.data());
    Falcons::Falcons_decompress_LorenzoReg<float, 1>(conf, new_bytes.data(), new_size, new_out.data());
    check(old_out == new_out, "composed predictor reconstruction differs");
    for (size_t i = 0; i < n; ++i)
        check(std::abs(double(input[i]) - new_out[i]) <= conf.absErrorBound,
              "composed predictor error bound exceeded");
    std::printf("PASS composed Lorenzo predictors huffman=%zu rlefse=%zu\n", old_size, new_size);
}

template<class T, uint N>
static void test_nd(Falcons::Config conf) {
    conf.cmprAlgo = Falcons::ALGO_LORENZO_REG;
    conf.errorBoundMode = Falcons::EB_ABS;
    conf.absErrorBound = 0.01;
    conf.lorenzo = true;
    conf.lorenzo2 = true;
    conf.regression = false;
    conf.regression2 = false;
    std::vector<T> input(conf.num);
    for (size_t i = 0; i < conf.num; ++i)
        input[i] = static_cast<T>(20.0 * std::sin(i * 0.113) + double(i % 97));
    std::vector<std::shared_ptr<Falcons::concepts::PredictorInterface<T, N>>> predictors;
    predictors.push_back(std::make_shared<Falcons::LorenzoPredictor<T, N, 1>>(conf.absErrorBound));
    predictors.push_back(std::make_shared<Falcons::LorenzoPredictor<T, N, 2>>(conf.absErrorBound));
    auto reference = Falcons::make_compressor_falcons_generic<T, N>(
        Falcons::make_decomposition_blockwise<T, N>(conf, Falcons::ComposedPredictor<T, N>(predictors),
            Falcons::LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2)),
        Falcons::HuffmanEncoder<int>(), Falcons::Lossless_zstd());
    const size_t capacity = Falcons::buffer_bounds::compressed_bound<T>(conf.num, conf.size_est());
    std::vector<Falcons::uchar> old_bytes(capacity), new_bytes(capacity);
    auto old_data = input, new_data = input;
    const size_t old_size = reference->compress(conf, old_data.data(), old_bytes.data(), capacity);
    const size_t new_size = Falcons::Falcons_compress_LorenzoReg<T, N>(conf, new_data.data(), new_bytes.data(), capacity);
    std::vector<T> old_out(conf.num), new_out(conf.num);
    reference->decompress(conf, old_bytes.data(), old_size, old_out.data());
    Falcons::Falcons_decompress_LorenzoReg<T, N>(conf, new_bytes.data(), new_size, new_out.data());
    check(old_out == new_out, "multidimensional Lorenzo reconstruction differs");
    for (size_t i = 0; i < conf.num; ++i)
        check(std::abs(double(input[i]) - double(new_out[i])) <= conf.absErrorBound,
              "multidimensional Lorenzo error bound exceeded");
    std::printf("PASS N=%u sizeof(T)=%zu huffman=%zu rlefse=%zu\n", N, sizeof(T), old_size, new_size);
}

int main() {
    try {
        for (int kind = 0; kind < 3; ++kind) run_case(kind);
        run_case(1, 16384);
        test_composed_predictor();
        test_nd<float, 2>(Falcons::Config(96, 64));
        test_nd<double, 3>(Falcons::Config(33, 34, 35));
        Falcons::Config conf(8);
        bool rejected = false;
        try {
            (void)Falcons::make_decomposition_blockwise<float, 1, int16_t>(
                conf, Falcons::LorenzoPredictor<float, 1, 1>(0.01), Falcons::LinearQuantizer<float>(0.01, 65536));
        } catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "oversized int16 quantizer radius was not rejected");
        std::puts("PASS invalid quantizer radius");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
