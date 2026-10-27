#ifndef Falcons_BYPASS_ENCODER_HPP
#define Falcons_BYPASS_ENCODER_HPP

#include <cassert>
#include <vector>
#include <memory>

#include "Encoder.hpp"
#include "Falcons/def.hpp"

namespace Falcons {

template <class T>
class BypassEncoder : public concepts::EncoderInterface<T> {
   public:
    void preprocess_encode(const std::vector<T> &bins, int stateNum) override {
    }

    size_t encode(const std::vector<T> &bins, uchar *&bytes) override {
        memcpy(bytes, &bins[0], sizeof(T) * bins.size());
        bytes += sizeof(T) * bins.size();
        return 0;
    }

    void postprocess_encode() override {}

    void preprocess_decode() override {}

    T* decode(const uchar *&bytes, size_t targetLength) override {
        std::unique_ptr<T[]> bins(new T[targetLength + 64 / sizeof(T)]{});
        memcpy(bins.get(), bytes, sizeof(T) * targetLength);
        bytes += sizeof(T) * targetLength;
        return bins.release();
    }

    void postprocess_decode() override {}

    void save(uchar *&c) override {}

    void load(const uchar *&c, size_t &remaining_length) override {}
};
}  // namespace Falcons
#endif
