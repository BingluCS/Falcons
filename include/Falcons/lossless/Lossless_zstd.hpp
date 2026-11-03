#ifndef Falcons_LOSSLESS_ZSTD_HPP
#define Falcons_LOSSLESS_ZSTD_HPP

#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>

#include "Falcons/def.hpp"
#include "Falcons/lossless/Lossless.hpp"
#include "Falcons/utils/MemoryUtil.hpp"
#include "zstd.h"
#include "zstd_errors.h"

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

    bool compress_bounded_trial(const uchar *src, size_t srcLen, uchar *dst, size_t dstCap,
                                size_t &written) {
        if (dstCap < sizeof(size_t)) return false;
        write(srcLen, dst);
        dstCap -= sizeof(size_t);
        if (std::getenv("FALCONS_NOZSTD")) {
            if (srcLen > dstCap) return false;
            if (srcLen) memcpy(dst, src, srcLen);
            written = srcLen + sizeof(size_t);
            return true;
        }
        if (!dstCap) return false;
        std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> ctx(ZSTD_createCCtx(), &ZSTD_freeCCtx);
        if (!ctx) throw std::bad_alloc();
        auto check = [](size_t result) {
            if (ZSTD_isError(result)) throw std::runtime_error(ZSTD_getErrorName(result));
        };
        check(ZSTD_CCtx_setParameter(ctx.get(), ZSTD_c_compressionLevel, compression_level));
        check(ZSTD_CCtx_setPledgedSrcSize(ctx.get(), srcLen));
        ZSTD_outBuffer out{dst, dstCap, 0};
        const size_t firstLen = srcLen ? srcLen - 1 : 0;
        ZSTD_inBuffer in{src, firstLen, 0};
        auto step = [&](ZSTD_EndDirective directive, size_t &remaining) {
            const size_t oldIn = in.pos, oldOut = out.pos;
            remaining = ZSTD_compressStream2(ctx.get(), &out, &in, directive);
            if (out.pos > out.size || in.pos > in.size) throw std::runtime_error("invalid Zstd stream position");
            if (ZSTD_isError(remaining)) {
                if (ZSTD_getErrorCode(remaining) == ZSTD_error_dstSize_tooSmall && out.pos == out.size) return false;
                throw std::runtime_error(ZSTD_getErrorName(remaining));
            }
            if (oldIn == in.pos && oldOut == out.pos && remaining && out.pos < out.size)
                throw std::runtime_error("Zstd stream made no progress");
            return true;
        };
        size_t remaining = 0;
        do {
            if (!step(ZSTD_e_continue, remaining)) return false;
            if (in.pos < in.size && out.pos == out.size) return false;
        } while (in.pos < in.size);
        in = {srcLen ? src + firstLen : src, srcLen - firstLen, 0};
        do {
            if (!step(ZSTD_e_end, remaining)) return false;
            if (remaining && out.pos == out.size) return false;
        } while (remaining);
        if (in.pos != in.size) throw std::runtime_error("Zstd stream did not consume its input");
        written = out.pos + sizeof(size_t);
        return true;
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
