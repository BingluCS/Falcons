#ifndef Falcons_LOSSLESS_ZSTD_HPP
#define Falcons_LOSSLESS_ZSTD_HPP

#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include "Falcons/def.hpp"
#include "Falcons/lossless/Lossless.hpp"
#include "Falcons/utils/MemoryUtil.hpp"
#include "zstd.h"

namespace Falcons {
class Lossless_zstd : public concepts::LosslessInterface {
   public:
    Lossless_zstd() = default;

    Lossless_zstd(int comp_level) : compression_level(comp_level) {}

    /**
     * Attention
     * When dstCap is smaller than the space needed, ZSTD will not throw any errors.
     * Instead, it will write a portion of the compressed data to dst and stops.
     * This behavior is not desirable in Falcons, as we need the whole compressed data for decompression.
     * Therefore, we need to check if the dst buffer (dstCap) is large enough for zstd
     */
    size_t compress(const uchar *src, size_t srcLen, uchar *dst, size_t dstCap) override {
        if (dstCap < sizeof(size_t))
            throw std::length_error(Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH);
        write(srcLen, dst);
        dstCap -= sizeof(size_t);  // reserve space for srcLen
        if (std::getenv("FALCONS_NOZSTD")) {                 // benchmark: skip zstd, store raw
            if (dstCap < srcLen) throw std::length_error(Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH);
            memcpy(dst, src, srcLen);
            return srcLen + sizeof(size_t);
        }
        if (dstCap < ZSTD_compressBound(srcLen)) {
            throw std::length_error(Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH);
        }
        size_t dstLen = ZSTD_compress(dst, dstCap, src, srcLen, compression_level);
        if (ZSTD_isError(dstLen))
            throw std::runtime_error(std::string("Zstd compression failed: ") + ZSTD_getErrorName(dstLen));
        return dstLen + sizeof(size_t);
    }

    size_t decompress(const uchar *src, const size_t srcLen, uchar *&dst, size_t &dstLen) override {
        if (src == nullptr || srcLen < sizeof(size_t))
            throw std::invalid_argument("Truncated Falcons lossless payload");
        read(dstLen, src);
        const size_t payloadSize = srcLen - sizeof(size_t);
        const bool raw = std::getenv("FALCONS_NOZSTD") != nullptr;
        if (raw) {
            if (dstLen > payloadSize)
                throw std::invalid_argument("Truncated Falcons raw payload");
        } else {
            const auto frameSize = ZSTD_getFrameContentSize(src, payloadSize);
            if (frameSize == ZSTD_CONTENTSIZE_ERROR ||
                (frameSize != ZSTD_CONTENTSIZE_UNKNOWN && frameSize != dstLen))
                throw std::invalid_argument("Invalid Falcons Zstd payload size");
        }
        const bool allocated = dst == nullptr;
        if (allocated) {
            dst = static_cast<uchar *>(malloc(dstLen == 0 ? 1 : dstLen));
            if (dst == nullptr) throw std::bad_alloc();
        }
        if (raw) {
            if (dstLen != 0) memcpy(dst, src, dstLen);
            return dstLen;
        }
        const size_t result = ZSTD_decompress(dst, dstLen, src, payloadSize);
        if (ZSTD_isError(result) || result != dstLen) {
            if (allocated) { free(dst); dst = nullptr; }
            throw std::invalid_argument("Invalid or truncated Falcons Zstd payload");
        }
        return result;
    }

   private:
    int compression_level = 3;  // default setting of level is 3
};
}  // namespace Falcons
#endif  // Falcons_LOSSLESS_ZSTD_HPP
