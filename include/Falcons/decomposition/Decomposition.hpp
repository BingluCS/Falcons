#ifndef Falcons_DECOMPOSITION_INTERFACE
#define Falcons_DECOMPOSITION_INTERFACE

#include <cstdint>
#include <vector>

#include "Falcons/def.hpp"
#include "Falcons/utils/Config.hpp"

namespace Falcons::concepts {
/**
 * Decomposition represents the lossy part in compressors (which typically converts original data to integers, e.g.,
 * prediction + quantization). Note: Falcons has both Predictor and Decomposition interfaces. You should choose Predictor
 * interface only when the new prediction method takes scalar value. In all other cases, use this Decomposition
 * interface.
 * @tparam Ti original data type
 * @tparam To decomposed data type
 * @tparam N original data dimension
 */
template <class Ti, class To, uint N, class ToAllocator = std::allocator<To>>
class DecompositionInterface {
   public:
    virtual ~DecompositionInterface() = default;

    /**
     * predict the data and quantize the error
     * @param data original input
     * @return quantized prediction error
     */
    virtual std::tuple<To*, uint64_t> compress(const Config &conf, Ti *data) = 0;

    /**
     * reverse of compress(), reconstruct the data
     * @param quant_inds quantized prediction error
     * @param dec_data place to write the reconstructed data
     * @return same value with dec_data
     */
    virtual Ti *decompress(const Config &conf, To* quant_inds, Ti *dec_data) = 0;

    /**
     * serialize the frontend and store it to a buffer
     * @param c One large buffer is pre-allocated, and the start location of the serialized frontend in the buffer is
     * indicated by c. After saving the frontend to the buffer, this function should change c to indicate the next empty
     * location in the buffer
     */
    virtual void save(uchar *&c) = 0;

    /**
     * deserialize the frontend from a buffer
     * @param c start location of the frontend in the buffer
     * @param remaining_length the remaining length of the buffer
     */
    virtual void load(const uchar *&c, size_t &remaining_length) = 0;

    virtual size_t size_est() { return 0; }

    virtual std::pair<int, int> get_out_range() = 0;

    virtual void print() {}

    //        virtual void clear() {};
};

template <class Ti, class To, uint N, class ToAllocator = std::allocator<To>>
class DecompositionInterface_OMP {
   public:
    virtual ~DecompositionInterface_OMP() = default;

    /**
     * predict the data and quantize the error
     * @param data original input
     * @return quantized prediction error
     */
    virtual std::tuple<To**, CacheLineInt*> compress(const Config &conf, Ti *data) = 0;

    /**
     * reverse of compress(), reconstruct the data
     * @param quant_inds quantized prediction error
     * @param dec_data place to write the reconstructed data
     * @return same value with dec_data
     */
    virtual Ti *decompress(const Config &conf, To** local_quant_inds, Ti *dec_data) = 0;

    /**
     * serialize the frontend and store it to a buffer
     * @param c One large buffer is pre-allocated, and the start location of the serialized frontend in the buffer is
     * indicated by c. After saving the frontend to the buffer, this function should change c to indicate the next empty
     * location in the buffer
     */
    virtual void save(uchar *&c) = 0;

    /**
     * deserialize the frontend from a buffer
     * @param c start location of the frontend in the buffer
     * @param remaining_length the remaining length of the buffer
     */
    virtual void load(const uchar *&c, size_t &remaining_length) = 0;

    virtual size_t size_est() { return 0; }

    virtual std::pair<int, int> get_out_range() = 0;

    virtual void print() {}

    //        virtual void clear() {};
};

}  // namespace Falcons::concepts

#endif
