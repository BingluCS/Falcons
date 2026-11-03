/**
 * @file FalconsGenericCompressor_Omp.hpp
 * @author Bing Lu
 * @brief Falcons parallel compression pipeline.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_COMPRESSOR_TYPE_ONE_OMP_HPP
#define Falcons_COMPRESSOR_TYPE_ONE_OMP_HPP

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "Falcons/compressor/Compressor.hpp"
#include "Falcons/decomposition/Decomposition.hpp"
#include "Falcons/def.hpp"
#include "Falcons/encoder/Encoder.hpp"
#include "Falcons/lossless/Lossless.hpp"
#include "Falcons/utils/BufferBounds.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/utils/FileUtil.hpp"
#include "Falcons/utils/Timer.hpp"

namespace Falcons {
/**
 * FalconsGenericCompressor glues together decomposition, encoder, and lossless modules to form the compressor.
 * It only takes Decomposition, not Predictor.
 * @tparam T original data type
 * @tparam N original data dimension
 * @tparam Decomposition decomposition module
 * @tparam Encoder encoder module
 * @tparam Lossless lossless module
 */

template <class T, uint N, class Decomposition, class Encoder, class Lossless>
class FalconsGenericCompressor_OMP : public concepts::CompressorInterface<T> {
    struct TemporaryStorageGuard {
        Decomposition &decomposition;
        ~TemporaryStorageGuard() { decomposition.release_temporary_storage(); }
    };

   public:
    FalconsGenericCompressor_OMP(Decomposition decomposition, Encoder encoder, Lossless lossless)
        : decomposition(std::move(decomposition)), encoder(std::move(encoder)), lossless(std::move(lossless)) {
        using QuantT = std::remove_pointer_t<std::remove_pointer_t<std::tuple_element_t<0,
            decltype(std::declval<Decomposition&>().compress(std::declval<const Config&>(), std::declval<T*>()))>>>;
        static_assert(std::is_base_of<concepts::DecompositionInterface_OMP<T, QuantT, N>, Decomposition>::value,
                      "must implement the frontend interface");
        static_assert(std::is_base_of<concepts::EncoderInterface<QuantT>, Encoder>::value,
                      "must implement the encoder interface");
        static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                      "must implement the lossless interface");
    }

    size_t compress(const Config &conf, T *data, uchar *cmpData, size_t cmpCap) override {
#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
        auto [local_quant_inds, local_quant_index] = decomposition.compress(conf, data);
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("cmp interp");
        timer.start();
#endif


        const int nthreads = decomposition.compression_thread_count();
        if (nthreads <= 0) throw std::runtime_error("Invalid decomposition thread count");
        const size_t header_size = buffer_bounds::add(
            sizeof(int), buffer_bounds::mul(static_cast<size_t>(nthreads), sizeof(size_t)));
        if (!cmpData || cmpCap < header_size)
            throw std::length_error(Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH);

        struct FreeBuffer { void operator()(uchar *p) const { std::free(p); } };
        using Buffer = std::unique_ptr<uchar, FreeBuffer>;
        std::vector<Buffer> blocks(nthreads);
        std::vector<size_t> block_sizes(nthreads);
        auto compress_block = [&](int tid, Encoder &cur_encoder) {
            const int quant_count = local_quant_index[tid].value;
            if (quant_count < 0) throw std::length_error("Invalid quantization count");
            const size_t scratch_size = buffer_bounds::scratch_bound<T>(
                decomposition.local_unpred_count(tid), static_cast<size_t>(quant_count));
            Buffer scratch(static_cast<uchar *>(std::malloc(scratch_size)));
            if (!scratch) throw std::bad_alloc();
            uchar *pos = scratch.get();
            if (tid == nthreads - 1) decomposition.save2(pos);
            decomposition.save3(pos, tid);
            if (nthreads == 1) write<size_t>(static_cast<size_t>(quant_count), pos);
            else write<int>(quant_count, pos);
            cur_encoder.preprocess_encode(local_quant_inds[tid], quant_count,
                                          decomposition.get_out_range().second);
            cur_encoder.save(pos);
            cur_encoder.encode(local_quant_inds[tid], quant_count, pos);
            cur_encoder.postprocess_encode();
            const size_t source_size = static_cast<size_t>(pos - scratch.get());
            const size_t zstd_bound = ZSTD_compressBound(source_size);
            if (ZSTD_isError(zstd_bound)) throw std::length_error("Zstd capacity overflow");
            const size_t capacity = buffer_bounds::add(sizeof(size_t), std::max(zstd_bound, source_size));
            Buffer block(static_cast<uchar *>(std::malloc(capacity)));
            if (!block) throw std::bad_alloc();
            block_sizes[tid] = lossless.compress(scratch.get(), source_size, block.get(), capacity);
            if (block_sizes[tid] > capacity) throw std::length_error("Invalid lossless output size");
            blocks[tid] = std::move(block);
        };

        if (nthreads == 1) {
            compress_block(0, encoder);
        } else {
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads)
            #endif
            for (int tid = 0; tid < nthreads; ++tid) {
                Encoder cur_encoder;
                compress_block(tid, cur_encoder);
            }
        }

        std::vector<size_t> offsets(nthreads);
        size_t payload_size = 0;
        for (int tid = 0; tid < nthreads; ++tid) {
            offsets[tid] = payload_size;
            payload_size = buffer_bounds::add(payload_size, block_sizes[tid]);
        }
        const size_t total_size = buffer_bounds::add(header_size, payload_size);
        if (total_size > cmpCap) throw std::length_error(Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH);
        uchar *cmpDataPos = cmpData;
        write<int>(nthreads, cmpDataPos);
        for (size_t offset : offsets) write<size_t>(offset, cmpDataPos);
        if (nthreads == 1) {
            write(blocks[0].get(), block_sizes[0], cmpDataPos);
        } else {
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads)
            #endif
            for (int tid = 0; tid < nthreads; ++tid) {
                std::memcpy(cmpDataPos + offsets[tid], blocks[tid].get(), block_sizes[tid]);
            }
            cmpDataPos += payload_size;
        }
#ifdef Falcons_PRINT_TIMINGS
        timer.stop("cmp all lossless");
#endif

        return cmpDataPos-cmpData;
    }

    T *decompress(const Config &conf, uchar const *cmpData, size_t cmpSize, T *decData) override {
        TemporaryStorageGuard cleanup{decomposition};
#ifdef Falcons_PRINT_TIMINGS
        Timer timer(true);
#endif
        if (!cmpData || cmpSize < sizeof(int))
            throw std::invalid_argument("Truncated OpenMP compression header");
        auto cmpDataPos = cmpData;
        int compression_thread_num = 0;
        read(compression_thread_num, cmpDataPos, cmpSize);
        if (compression_thread_num <= 0 ||
            static_cast<size_t>(compression_thread_num) > cmpSize / sizeof(size_t))
            throw std::invalid_argument("Invalid OpenMP compression thread count");
        if (conf.dims.size() != N) throw std::invalid_argument("Invalid interpolation dimensions");
        size_t data_count = 1;
        for (size_t dim : conf.dims) {
            if (dim == 0) throw std::invalid_argument("Invalid interpolation dimensions");
            data_count = buffer_bounds::mul(data_count, dim);
        }

        std::vector<size_t> offsets(compression_thread_num);
        read(offsets.data(), offsets.size(), cmpDataPos, cmpSize);
        if (offsets[0] != 0) throw std::invalid_argument("Invalid OpenMP block offset");
        for (int tid = 0; tid < compression_thread_num; ++tid) {
            if (offsets[tid] >= cmpSize || (tid > 0 && offsets[tid] <= offsets[tid - 1]))
                throw std::invalid_argument("Invalid OpenMP block offset");
        }

        struct FreeBuffer { void operator()(uchar *p) const { std::free(p); } };
        using Buffer = std::unique_ptr<uchar, FreeBuffer>;
        std::vector<Buffer> buffers(compression_thread_num);
        std::vector<const uchar *> positions(compression_thread_num);
        std::vector<size_t> remaining(compression_thread_num);
        std::vector<std::exception_ptr> errors(compression_thread_num);
        auto decompress_block = [&](int tid) {
            const size_t end = tid + 1 < compression_thread_num ? offsets[tid + 1] : cmpSize;
            uchar *buffer = nullptr;
            try {
                lossless.decompress(cmpDataPos + offsets[tid], end - offsets[tid], buffer, remaining[tid]);
            } catch (...) {
                std::free(buffer);
                throw;
            }
            buffers[tid].reset(buffer);
            positions[tid] = buffer;
        };
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(compression_thread_num)
        #endif
        for (int tid = 0; tid < compression_thread_num; ++tid) {
            try { decompress_block(tid); }
            catch (...) { errors[tid] = std::current_exception(); }
        }
        for (const auto &error : errors) if (error) std::rethrow_exception(error);

        const int last = compression_thread_num - 1;
        decomposition.load2(positions[last], remaining[last]);
        decomposition.init_local_unpred(compression_thread_num);
        std::vector<std::unique_ptr<int16_t[]>> owned_quant_inds(compression_thread_num);
        std::vector<int16_t *> local_quant_inds(compression_thread_num, nullptr);
        auto decode_block = [&](int tid, Encoder &cur_encoder) {
            decomposition.load3(positions[tid], remaining[tid], tid);
            size_t quant_count = 0;
            if (compression_thread_num == 1) {
                if (remaining[tid] < sizeof(size_t)) throw std::invalid_argument("Truncated quantization count");
                read(quant_count, positions[tid], remaining[tid]);
            } else {
                if (remaining[tid] < sizeof(int)) throw std::invalid_argument("Truncated quantization count");
                int count = 0;
                read(count, positions[tid], remaining[tid]);
                if (count < 0) throw std::invalid_argument("Invalid quantization count");
                quant_count = static_cast<size_t>(count);
            }
            if (quant_count > data_count || quant_count > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Invalid quantization count");
            cur_encoder.load(positions[tid], remaining[tid]);
            owned_quant_inds[tid].reset(cur_encoder.decode(positions[tid], quant_count));
            local_quant_inds[tid] = owned_quant_inds[tid].get();
            cur_encoder.postprocess_decode();
            buffers[tid].reset();
        };
        if (compression_thread_num == 1) {
            decode_block(0, encoder);
        } else {
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(compression_thread_num)
            #endif
            for (int tid = 0; tid < compression_thread_num; ++tid) {
                try {
                    Encoder cur_encoder;
                    decode_block(tid, cur_encoder);
                } catch (...) {
                    errors[tid] = std::current_exception();
                }
            }
            for (const auto &error : errors) if (error) std::rethrow_exception(error);
        }

#ifdef Falcons_PRINT_TIMINGS
        timer.stop("decmp all lossless");
        timer.start();
#endif
        decomposition.decompress(conf, local_quant_inds.data(), decData);
#ifdef Falcons_PRINT_TIMINGS
    timer.stop("decomp interp");
#endif
        return decData;
    }

   private:
    Decomposition decomposition;
    Encoder encoder;
    Lossless lossless;
};

template <class T, uint N, class Decomposition, class Encoder, class Lossless>
std::shared_ptr<FalconsGenericCompressor_OMP<T, N, Decomposition, Encoder, Lossless>> make_compressor_falcons_generic_omp(
    Decomposition decomposition, Encoder encoder, Lossless lossless) {
    return std::make_shared<FalconsGenericCompressor_OMP<T, N, Decomposition, Encoder, Lossless>>(
        std::move(decomposition), std::move(encoder), std::move(lossless));
}

}  // namespace Falcons
#endif
