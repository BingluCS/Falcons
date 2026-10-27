#ifndef Falcons_RUNLENGTH_ENCODER_HPP
#define Falcons_RUNLENGTH_ENCODER_HPP

#include <vector>
#include <memory>

#include "Encoder.hpp"
#include "Falcons/def.hpp"
#include "Falcons/utils/MemoryUtil.hpp"

namespace Falcons {

template <class T>
class RunlengthEncoder : public concepts::EncoderInterface<T> {
   public:
    void preprocess_encode(const std::vector<T> &bins, int stateNum) override {}

    size_t encode(const std::vector<T> &bins, uchar *&bytes) override {
        auto bytespos = bytes;
        int max = 0;
        size_t s = 0;
        for (size_t i = 1; i < bins.size(); i++) {
            if (bins[i] != bins[i - 1]) {
                write(bins[i - 1], bytes);
                write(int(i - s), bytes);
                if (int(i - s) > max) {
                    max = int(i - s);
                }
                s = i;
            }
        }
        write(bins[bins.size() - 1], bytes);
        write(int(bins.size() - s), bytes);
        return bytes - bytespos;
    }

    void postprocess_encode() override {}

    void preprocess_decode() override {}

    T* decode(const uchar *&bytes, size_t targetLength) override {
        std::unique_ptr<T[]> bins(new T[targetLength + 64 / sizeof(T)]{});
        T value;
        int cnt;
        for (size_t i = 0; i < targetLength;) {
            read(value, bytes);
            read(cnt, bytes);
            if (cnt <= 0 || static_cast<size_t>(cnt) > targetLength - i) {
                throw std::runtime_error("Decoded length exceeds targetLength");
            }
            for (size_t j = i; j < i + cnt; j++) {
                bins[j] = value;
            }
            i += cnt;
        }
        return bins.release();
    }

    void postprocess_decode() override {}

    void save(uchar *&c) override {}

    void load(const uchar *&c, size_t &remaining_length) override {}
};
}  // namespace Falcons
#endif
