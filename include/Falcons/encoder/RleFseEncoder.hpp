/**
 * @file RleFseEncoder.hpp
 * @author Bing Lu
 * @brief Adaptive run-length and FSE encoding for Falcons.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_RLE_FSE_ENCODER_HPP
#define Falcons_RLE_FSE_ENCODER_HPP

/**
 * Adaptive encoding for signed 16-bit quantization codes:
 * RLE+FSE for sparse codes, conditional tANS for denser codes, and
 * four-stream byte-split FSE for large streams with many escapes.
 * The caller applies the final lossless stage separately.
 */

#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <cstring>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#endif

#include "Falcons/def.hpp"
#include "Falcons/encoder/Encoder.hpp"
#include "Falcons/utils/MemoryUtil.hpp"

#define FSE_STATIC_LINKING_ONLY
#include "zfse/fse.h"
extern "C" {
#include "zfse/hist.h"
size_t FSE_buildCTable(FSE_CTable *ct, const short *normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
size_t FSE_buildDTable(FSE_DTable *dt, const short *normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
}

namespace Falcons {

template <class T>
class RleFseEncoder : public concepts::EncoderInterface<T> {
    static_assert(sizeof(T) == 2, "RleFseEncoder expects 16-bit quant codes");

    // FSE blocks use a table or a RAW/RLE fallback, with at most 64K bytes.
    static constexpr int FBLK = 1 << 16;

    uint16_t lo_ = 0, mode_ = 0;
    uint8_t method_ = 0;                         // 0=RLE+FSE, 1=tANS, 2=byte-split FSE
    bool use_delta_ = false;                     // forward-diff delta (decided by dzeros)
    bool scan_simd_ = true;                      // SIMD in scan_run (decided by p0>=0.85)
    size_t decode_remaining_ = 0;

    // void derive(const T *bins, size_t num_bin, int stateNum) {
    //     (void)stateNum;
    //     mode_ = 0; lo_ = 0;
    //     // adaptive backend selection by p(mode):
    //     //   p0 >= thr (0.80) -> RLE+FSE (method 0)  : loose/sparse, long mode runs
    //     //   p0 <  thr        -> tANS    (method 1)  : tight/dense, order-1 residual structure
    //     size_t S = num_bin < 4096 ? num_bin : 4096; 
    //     size_t stride = (S > 1) ? num_bin / S : 1; 
    //     if (!stride) 
    //         stride = 1;
    //     size_t hit = 0; 
    //     for (size_t k = 0; k < S; k++) 
    //         hit += (bins[k * stride] == mode_);
    //     double p0 = S ? (double)hit / (double)S : 1.0;
    //     double thr = 0.80; 
    //     // if (const char *e = std::getenv("FALCONS_P0THR")) 
    //     //     thr = atof(e);
    //     method_ = (p0 >= thr) ? 0 : 1;
    //     // if (const char *e = std::getenv("FALCONS_FORCE")) method_ = (uint8_t)atoi(e);   // force backend: 0=RLE+FSE 1=tANS 2=RLE+ADT2 3=ADT2 4=rANS 5=demux-ADT
    //     // if (std::getenv("FALCONS_DUMP_METRICS")) std::fprintf(stderr, "[QM] N=%zu p0=%.5f thr=%.2f method=%d\n", num_bin, p0, thr, method_);
    // }

    static inline size_t scan_run(const T *a, size_t i, size_t n, T mode, bool use_simd) {
        size_t s = i;
#ifdef __AVX2__
        if (use_simd) {                                  // p0>=0.85 (long runs) -> SIMD; else scalar
            const __m256i vm = _mm256_set1_epi16(static_cast<int16_t>(mode));
            while (i + 16 <= n) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
                uint32_t m = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, vm)));
                if (m != 0xFFFFFFFFu) return i + (__builtin_ctz(~m) >> 1) - s;
                i += 16;
            }
        }
#elif defined(__ARM_FEATURE_SVE2)
        if (use_simd) {
            const size_t lanes = svcnth();
            const svbool_t pg = svptrue_b16();
            const int16_t mode_s16 = static_cast<int16_t>(mode);
            while (i + lanes <= n) {
                const svint16_t values =
                    svld1_s16(pg, reinterpret_cast<const int16_t *>(a + i));
                const svbool_t mismatch = svcmpne_n_s16(pg, values, mode_s16);
                if (svptest_any(pg, mismatch)) {
                    const svbool_t before_first = svbrkb_z(pg, mismatch);
                    return i + svcntp_b16(pg, before_first) - s;
                }
                i += lanes;
            }
        }
#endif
        while (i < n && a[i] == mode) i++;
        return i - s;
    }

#ifdef __AVX2__
    static inline size_t rle_scan_run_cached(const T *a, size_t i, size_t n, T mode,
                                             __m256i vm, size_t &block_end,
                                             uint32_t &nonmode_mask) {
        const size_t start = i;
        // Consume each block's mismatches once, with one mask bit per 16-bit lane.
        while (!nonmode_mask && n - block_end >= 16) {
            const __m256i values =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + block_end));
            nonmode_mask = ~static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi16(values, vm))) & 0x55555555u;
            block_end += 16;
        }
        if (nonmode_mask) {
            const size_t next = block_end - 16 + (__builtin_ctz(nonmode_mask) >> 1);
            nonmode_mask &= nonmode_mask - 1;
            return next - start;
        }
        i = std::max(i, block_end);
        while (i < n && a[i] == mode) ++i;
        return i - start;
    }
#endif

    static inline void rle_flush_value_mantissa(uint8_t *dst, size_t capacity, size_t &pos,
                                               uint64_t acc, int &bits) {
        // At most 17 value bits plus 7 carry bits; only complete bytes advance pos.
        if (pos <= capacity && capacity - pos >= sizeof(uint32_t)) {
            const uint32_t word = static_cast<uint32_t>(
                acc << ((32u - static_cast<unsigned>(bits)) & 31u));
            const uint8_t packed[4] = {
                static_cast<uint8_t>(word >> 24), static_cast<uint8_t>(word >> 16),
                static_cast<uint8_t>(word >> 8), static_cast<uint8_t>(word)
            };
            memcpy(dst + pos, packed, sizeof(packed));
            pos += static_cast<unsigned>(bits) >> 3;
            bits &= 7;
            return;
        }
        while (bits >= 8) {
            bits -= 8;
            dst[pos++] = static_cast<uint8_t>(acc >> bits);
        }
    }

    // Write L copies of mode (decode mirror of scan_run's SIMD find).
    static inline void fill_mode(T *dst, size_t L, T mode) {
// #ifdef __AVX2__
//         const __m256i vm = _mm256_set1_epi16(static_cast<int16_t>(mode));
//         size_t j = 0;
//         for (; j + 16 <= L; j += 16) _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + j), vm);
//         for (; j < L; j++) dst[j] = mode;
// #else
        std::fill_n(dst, L, mode);
// #endif
    }

    // FSE class bytes in blocks: [u8 flag][u32 clen?][data]; returns bytes written
    size_t fse_enc(const uint8_t *src, size_t n, uint8_t *out) {
        if (n == 0) return 0;
        size_t p = 0; 
        uint8_t *tmp = encode_workspace().fse_tmp.get(FBLK + 1024);
        for (size_t off = 0; off < n; off += FBLK) {
            size_t len = n - off < FBLK ? n - off : FBLK;
            size_t r = FSE_compress(tmp, FBLK + 1024, src + off, len);
            if (FSE_isError(r) || r == 0 || r >= len) { 
                out[p++] = 0; 
                memcpy(out + p, src + off, len); 
                p += len;
            }
            else if (r == 1) { 
                out[p++] = 2; 
                out[p++] = src[off]; 
            }   // FSE RLE: block is one repeated symbol (FSE_decompress can't decode r==1)
            else { 
                out[p++] = 1; 
                uint32_t rr = static_cast<uint32_t>(r);
                memcpy(out + p, &rr, 4);
                p += 4;
                memcpy(out + p, tmp, r); 
                p += r; 
            }
        }
        return p;
    }
    static void fse_dec(const uint8_t *in, size_t n_orig, uint8_t *dst) {
        size_t dp = 0, rp = 0;
        while (dp < n_orig) {
            size_t len = n_orig - dp < static_cast<size_t>(FBLK) ? n_orig - dp : static_cast<size_t>(FBLK);
            uint8_t flag = in[rp++];
            if (flag == 0) { memcpy(dst + dp, in + rp, len); rp += len; }
            else if (flag == 2) { memset(dst + dp, in[rp++], len); }   // FSE RLE
            else { uint32_t rr; memcpy(&rr, in + rp, 4); rp += 4; FSE_decompress(dst + dp, len, in + rp, rr); rp += rr; }
            dp += len;
        }
    }

    // ---- order-1 model constants (context buckets + tANS byte alphabet) ----
    static constexpr int O1_NCTX = 15;
    static constexpr int O1T_PB = 13;
    static constexpr uint32_t O1T_M = 1u << O1T_PB;
    static constexpr int O1T_NSYM = 256;               // tANS: FSE byte alphabet; 255 = escape (|code|>=128)
    static inline int o1_unzz(uint32_t z) { return ((z >> 1) ^ (~(z & 1) + 1)); }   // inverse zigzag (decode side)

    // Precomputed LUTs over all 65536 int16 codes, for branchless O(1) lookup in the tANS hot loop:
    //   ctx[code]    = context bucket of prev code (0 for 0; fine for |v|<=Bf, geometric beyond; x2 for sign).
    //   sym256[code] = zigzag(code) clamped to [0,254]; 255 = escape (|code|>=128, raw value to side stream).
    // The constructor fills both tables, so o1_lut() is a one-line, build-once, thread-safe singleton.
    struct O1Lut {
        uint8_t ctx[65536];
        uint8_t sym256[65536];
        O1Lut() {
            constexpr int B = (O1_NCTX - 1) / 2, Bf = (B + 1) / 2;   // 15 buckets/sign, 8 fine
            for (int u = 0; u < 65536; u++) {
                int ctxv, code = static_cast<int16_t>(u);
                int sgn = code < 0;
                int a = sgn ? -code : code;     // --- ctx: magnitude+sign bucket ---
                if (a == 0) 
                    ctxv = 0;
                else {
                    int m;
                    if (a <= Bf) 
                        m = a - 1;
                    else { 
                        m = Bf - 1; 
                        int v = Bf; 
                        while (v < a && m < B - 1) { 
                            v += (v >> 1) + 1; 
                            m++; 
                        } 
                    }
                    if (m >= B) 
                        m = B - 1;
                    ctxv = 1 + m * 2 + sgn;
                }
                ctx[u] = ctxv;
                uint32_t zz = (static_cast<uint32_t>(code) << 1) ^ (code >> 31);   // --- sym256: zigzag, 255=escape ---
                sym256[u] = static_cast<uint8_t>(zz < O1T_NSYM - 1 ? zz : (O1T_NSYM - 1));
            }
        }
    };

    static const O1Lut &o1_lut() { 
        static const O1Lut t; 
        return t; 
    }

    static void o1_normalize(const uint32_t *cnt, uint16_t *freq, int nsym = O1T_NSYM, int Mval = O1T_M) {
        uint64_t total = 0; 
        int nonzero = 0;
        for (int s = 0; s < nsym; s++) {
            total += cnt[s];
            nonzero += (cnt[s] != 0);
        }
        if (total == 0) { 
            for (int s = 0; s < nsym; s++) 
            freq[s] = 0; 
            freq[0] = Mval; 
            return; 
        }
        if (nonzero > 1 && Mval > 0 && (Mval & (Mval - 1)) == 0) {
            int16_t norm[O1T_NSYM];
            unsigned tableLog = 31u - static_cast<unsigned>(__builtin_clz(static_cast<unsigned>(Mval)));
            size_t r = FSE_normalizeCount(norm, tableLog, cnt, static_cast<size_t>(total),
                                          static_cast<unsigned>(nsym - 1));
            if (!FSE_isError(r)) {
                for (int s = 0; s < nsym; s++)
                    freq[s] = static_cast<uint16_t>(norm[s]);
                return;
            }
        }
        int sum = 0, maxf = 0; 
        int argmax = 0;
        for (int s = 0; s < nsym; s++) {
            if (cnt[s]) { 
                int f = static_cast<uint64_t>(cnt[s]) * Mval / total; 
                if (f == 0) 
                    f = 1;
                freq[s] = static_cast<uint16_t>(f); 
                sum += f; 
                if (f > maxf) {
                    maxf = f; 
                    argmax = s; 
                } 
            }
            else freq[s] = 0;
        }
        // make context total exactly Mval, robustly (flat contexts: argmax can be small, so spread the cut)
        int e = Mval - sum;
        if (e > 0) { 
            freq[argmax] = static_cast<int>(freq[argmax]) + e; 
        }
        else if (e < 0) {
            e = -e;
            int take = (maxf - 1 < e) ? (maxf - 1) : e; 
            freq[argmax] = static_cast<int>(freq[argmax]) - take;
            e -= take;
            while (e > 0) {                              
                int mi = -1; 
                uint32_t mv = 1;
                for (int s = 0; s < nsym; s++) 
                    if (freq[s] > mv) { 
                        mv = freq[s]; 
                        mi = s; 
                    }
                if (mi < 0) 
                    break;                               // nothing reducible (shouldn't happen)
                take = static_cast<int>(freq[mi]) - 1; 
                if (take > e) 
                    take = e;
                freq[mi] = static_cast<int>(freq[mi]) - take; 
                e -= take;
            }
        }
    }

    static constexpr int O1T_K = 4;   // tANS: number of interleaved bitstreams for ILP (max 16; arrays sized 16)

    template <class V>
    struct EncodeBuffer {
        std::unique_ptr<V[]> data;
        size_t capacity = 0;
        V *get(size_t n) {
            if (n > capacity) {
                data.reset(new V[n]);
                capacity = n;
            }
            return data.get();
        }
    };

    static constexpr unsigned BYTE_FSE_TABLE_LOG = 11;
    static constexpr size_t BYTE_FSE_STREAMS = 4;
    static constexpr size_t BYTE_FSE_LENGTHS = 2 * BYTE_FSE_STREAMS;

    struct ByteFseWorkspace {
        unsigned count[256];
        short normalized[256];
        FSE_CTable ctable[1537];
        FSE_DTable dtable[2049];
        unsigned scratch[1024];
        uint8_t header[512];
        uint8_t streams[BYTE_FSE_STREAMS][22560];
    };
    static_assert(sizeof(unsigned) == 4 && sizeof(short) == 2, "FSE ABI mismatch");
    static_assert(sizeof(size_t) == 4 || sizeof(size_t) == 8, "unsupported FSE bit container size");
    static_assert(sizeof(ByteFseWorkspace::ctable) >= FSE_CTABLE_SIZE(BYTE_FSE_TABLE_LOG, 255));
    static_assert(sizeof(ByteFseWorkspace::dtable) >= FSE_DTABLE_SIZE(BYTE_FSE_TABLE_LOG));
    static_assert(sizeof(ByteFseWorkspace::scratch) >= HIST_WKSP_SIZE);
    static_assert(sizeof(ByteFseWorkspace::header) >= FSE_NCOUNTBOUND);
    static_assert(sizeof(ByteFseWorkspace::streams[0]) >= (16384 * BYTE_FSE_TABLE_LOG + 7) / 8 + 16);

    static void check_byte_fse(size_t result) {
        if (FSE_isError(result)) throw std::runtime_error(FSE_getErrorName(result));
    }

    static size_t byte_split_bound(size_t count) {
        const size_t blocks = count / FBLK + (count % FBLK != 0);
        const size_t limit = static_cast<size_t>((std::numeric_limits<ptrdiff_t>::max)());
        if (blocks > limit / 2 || count > (limit - 2 * blocks) / 2)
            throw std::length_error("byte split output size overflow");
        return 2 * count + 2 * blocks;
    }

    static void split_bytes(const T *input, size_t count, uint8_t *low, uint8_t *high) {
        size_t i = 0;
#ifdef __AVX2__
        const __m256i mask = _mm256_set1_epi16(255);
        for (; count - i >= 16; i += 16) {
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
            const __m256i zz = _mm256_xor_si256(_mm256_slli_epi16(x, 1), _mm256_srai_epi16(x, 15));
            const __m256i l = _mm256_and_si256(zz, mask);
            const __m256i h = _mm256_srli_epi16(zz, 8);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(low + i),
                _mm_packus_epi16(_mm256_castsi256_si128(l), _mm256_extracti128_si256(l, 1)));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(high + i),
                _mm_packus_epi16(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));
        }
#endif
        for (; i < count; ++i) {
            const uint16_t x = static_cast<uint16_t>(input[i]);
            const uint16_t zz = static_cast<uint16_t>((static_cast<unsigned>(x) << 1) ^ (0u - (x >> 15)));
            low[i] = static_cast<uint8_t>(zz);
            high[i] = static_cast<uint8_t>(zz >> 8);
        }
    }

    static void join_bytes(const uint8_t *low, const uint8_t *high, size_t count, T *output) {
        size_t i = 0;
#ifdef __AVX2__
        const __m256i one = _mm256_set1_epi16(1), zero = _mm256_setzero_si256();
        for (; count - i >= 16; i += 16) {
            const __m128i l = _mm_loadu_si128(reinterpret_cast<const __m128i *>(low + i));
            const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(high + i));
            const __m256i zz = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_unpacklo_epi8(l, h)), _mm_unpackhi_epi8(l, h), 1);
            const __m256i x = _mm256_xor_si256(_mm256_srli_epi16(zz, 1),
                _mm256_sub_epi16(zero, _mm256_and_si256(zz, one)));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), x);
        }
#endif
        for (; i < count; ++i) {
            const uint16_t zz = static_cast<uint16_t>(low[i] | (unsigned(high[i]) << 8));
            const uint16_t x = static_cast<uint16_t>((zz >> 1) ^ (0u - (zz & 1u)));
            output[i] = static_cast<T>(x);
        }
    }

    // Each byte block is stored as RAW (0), FSE (1), or RLE (2).
    static void encode_byte_stream(const uint8_t *src, size_t n, uint8_t *&out,
                                   ByteFseWorkspace &ws) {
        auto raw = [&] {
            *out++ = 0;
            std::memcpy(out, src, n);
            out += n;
        };
        if (n < BYTE_FSE_STREAMS) {
            if (n > 1 && std::all_of(src + 1, src + n, [src](uint8_t b) { return b == src[0]; })) {
                *out++ = 2; *out++ = src[0];
            } else raw();
            return;
        }

        unsigned max_symbol = 255;
        const size_t max_count = HIST_count_wksp(ws.count, &max_symbol, src, n,
                                                  ws.scratch, sizeof(ws.scratch));
        if (HIST_isError(max_count)) throw std::runtime_error("HIST_count_wksp failed");
        if (max_count == n) { *out++ = 2; *out++ = src[0]; return; }
        if (max_count == 1 || max_count < (n >> 7)) { raw(); return; }

        const unsigned table_log = FSE_optimalTableLog(BYTE_FSE_TABLE_LOG, n, max_symbol);
        const size_t normalized = FSE_normalizeCount(ws.normalized, table_log, ws.count, n, max_symbol);
        check_byte_fse(normalized);
        if (normalized != table_log) throw std::runtime_error("unexpected FSE normalization");
        const size_t header_size = FSE_writeNCount(ws.header, sizeof(ws.header), ws.normalized,
                                                   max_symbol, table_log);
        check_byte_fse(header_size);
        check_byte_fse(FSE_buildCTable_wksp(ws.ctable, ws.normalized, max_symbol, table_log,
                                            ws.scratch, sizeof(ws.scratch)));

        BIT_CStream_t bits[BYTE_FSE_STREAMS];
        FSE_CState_t states[BYTE_FSE_STREAMS];
        size_t offsets[BYTE_FSE_STREAMS];
        const size_t quarter = n / BYTE_FSE_STREAMS, remainder = n % BYTE_FSE_STREAMS;
        for (size_t i = 0; i < BYTE_FSE_STREAMS; ++i) {
            offsets[i] = i * quarter + (std::min)(i, remainder);
            check_byte_fse(BIT_initCStream(&bits[i], ws.streams[i], sizeof(ws.streams[i])));
            FSE_initCState(&states[i], ws.ctable);
            if (i < remainder) {
                FSE_encodeSymbol(&bits[i], &states[i], src[offsets[i] + quarter]);
                BIT_flushBits(&bits[i]);
            }
        }
        auto encode_at = [&](size_t p) {
            FSE_encodeSymbol(&bits[0], &states[0], src[offsets[0] + p]);
            FSE_encodeSymbol(&bits[1], &states[1], src[offsets[1] + p]);
            FSE_encodeSymbol(&bits[2], &states[2], src[offsets[2] + p]);
            FSE_encodeSymbol(&bits[3], &states[3], src[offsets[3] + p]);
        };
        auto flush = [&] {
            for (size_t i = 0; i < BYTE_FSE_STREAMS; ++i) BIT_flushBits(&bits[i]);
        };
        size_t p = quarter;
        while (p >= 4) {
            encode_at(--p); encode_at(--p);
            // On 32-bit, two 11-bit symbols plus at most seven pending bits fit in size_t.
            if constexpr (sizeof(size_t) == 4) flush();
            encode_at(--p); encode_at(--p);
            flush();
        }
        while (p) {
            encode_at(--p);
            if constexpr (sizeof(size_t) == 4) flush();
        }

        size_t lengths[BYTE_FSE_STREAMS], frame_size = BYTE_FSE_LENGTHS + header_size;
        for (size_t i = 0; i < BYTE_FSE_STREAMS; ++i) {
            FSE_flushCState(&bits[i], &states[i]);
            lengths[i] = BIT_closeCStream(&bits[i]);
            if (!lengths[i] || lengths[i] > 65535)
                throw std::runtime_error("FSE stream buffer exhausted");
            frame_size += lengths[i];
        }
        if (frame_size + sizeof(uint32_t) >= n) { raw(); return; }

        *out++ = 1;
        const uint32_t size = static_cast<uint32_t>(frame_size);
        std::memcpy(out, &size, sizeof(size)); out += sizeof(size);
        for (size_t i = 0; i < BYTE_FSE_STREAMS; ++i) {
            *out++ = static_cast<uint8_t>(lengths[i]);
            *out++ = static_cast<uint8_t>(lengths[i] >> 8);
        }
        std::memcpy(out, ws.header, header_size); out += header_size;
        for (size_t i = 0; i < BYTE_FSE_STREAMS; ++i) {
            std::memcpy(out, ws.streams[i], lengths[i]); out += lengths[i];
        }
    }

    struct ReverseBits {
        const uint8_t *data;
        size_t pos;
        uint64_t bits;
        unsigned available;

        ReverseBits(const uint8_t *p, size_t size) : data(p), pos(size - 1), bits(0), available(0) {
            const unsigned last = p[size - 1];
            if (!last) throw std::runtime_error("FSE stream has no end marker");
            for (unsigned value = last; value > 1; value >>= 1) ++available;
            bits = last & ((1u << available) - 1u);
        }
        bool ensure(unsigned needed) {
            if (available < needed && pos >= 8) {
                const unsigned take = (63u - available) / 8u;
                const uint64_t word = MEM_readLE64(data + pos - 8);
                bits = (bits << (take * 8)) | (word >> ((8 - take) * 8));
                pos -= take;
                available += take * 8;
            }
            while (available < needed && pos != 0) {
                bits = (bits << 8) | data[--pos];
                available += 8;
            }
            return available >= needed;
        }
        unsigned read_unchecked(unsigned count) {
            available -= count;
            return static_cast<unsigned>((bits >> available) & ((1u << count) - 1u));
        }
        unsigned read(unsigned count) {
            if (!ensure(count)) throw std::runtime_error("truncated FSE stream");
            return read_unchecked(count);
        }
        bool finished() const { return pos == 0 && available == 0; }
    };

    static void decode_byte_stream(const uint8_t *&in, size_t &remaining, size_t n, uint8_t *dst,
                                   ByteFseWorkspace &ws) {
        auto take = [&](size_t count) {
            if (count > remaining) throw std::runtime_error("truncated byte split stream");
            const uint8_t *p = in; in += count; remaining -= count; return p;
        };
        const uint8_t mode = *take(1);
        if (mode == 0) { std::memcpy(dst, take(n), n); return; }
        if (mode == 2) { std::memset(dst, *take(1), n); return; }
        if (mode != 1) throw std::runtime_error("invalid byte split mode");

        uint32_t frame_size;
        std::memcpy(&frame_size, take(sizeof(frame_size)), sizeof(frame_size));
        if (frame_size <= 1 || n <= sizeof(frame_size) || frame_size >= n - sizeof(frame_size))
            throw std::runtime_error("invalid byte split FSE length");
        const uint8_t *src = take(frame_size);
        if (n < BYTE_FSE_STREAMS || n > FBLK || frame_size < BYTE_FSE_LENGTHS + BYTE_FSE_STREAMS + 1)
            throw std::runtime_error("invalid FSE frame size");

        size_t lengths[BYTE_FSE_STREAMS], payload_size = 0;
        for (size_t i = 0; i < BYTE_FSE_STREAMS; ++i) {
            lengths[i] = src[2 * i] | (static_cast<size_t>(src[2 * i + 1]) << 8);
            if (!lengths[i] || lengths[i] > sizeof(ws.streams[i]))
                throw std::runtime_error("invalid FSE stream length");
            payload_size += lengths[i];
        }
        if (payload_size >= frame_size - BYTE_FSE_LENGTHS)
            throw std::runtime_error("truncated FSE frame");
        const size_t header_size = frame_size - BYTE_FSE_LENGTHS - payload_size;
        if (header_size > sizeof(ws.header) ||
            (src[BYTE_FSE_LENGTHS] & 15u) > BYTE_FSE_TABLE_LOG - FSE_MIN_TABLELOG)
            throw std::runtime_error("invalid FSE model header");
        unsigned max_symbol = 255, table_log = 0;
        const size_t read = FSE_readNCount(ws.normalized, &max_symbol, &table_log,
                                            src + BYTE_FSE_LENGTHS, frame_size - BYTE_FSE_LENGTHS);
        check_byte_fse(read);
        if (read != header_size || table_log < FSE_MIN_TABLELOG || table_log > BYTE_FSE_TABLE_LOG)
            throw std::runtime_error("invalid FSE model size or tableLog");
        unsigned total = 0, present = 0;
        for (unsigned i = 0; i <= max_symbol; ++i) {
            const int count = ws.normalized[i];
            if (count < -1) throw std::runtime_error("invalid FSE normalized count");
            if (count) ++present;
            total += static_cast<unsigned>(count < 0 ? -count : count);
        }
        if (total != (1u << table_log) || present < 2)
            throw std::runtime_error("invalid FSE normalized total");
        check_byte_fse(FSE_buildDTable(ws.dtable, ws.normalized, max_symbol, table_log));

        const uint8_t *payload = src + BYTE_FSE_LENGTHS + header_size;
        ReverseBits b0(payload, lengths[0]); payload += lengths[0];
        ReverseBits b1(payload, lengths[1]); payload += lengths[1];
        ReverseBits b2(payload, lengths[2]); payload += lengths[2];
        ReverseBits b3(payload, lengths[3]);
        ReverseBits *bits[BYTE_FSE_STREAMS] = {&b0, &b1, &b2, &b3};
        unsigned states[BYTE_FSE_STREAMS];
        const auto *table = reinterpret_cast<const FSE_decode_t *>(ws.dtable + 1);
        const unsigned table_size = 1u << table_log;
        for (unsigned s = 0; s < table_size; ++s) {
            const FSE_decode_t entry = table[s];
            if (entry.nbBits > table_log ||
                static_cast<unsigned>(entry.newState) + ((1u << entry.nbBits) - 1u) >= table_size)
                throw std::runtime_error("invalid FSE transition table");
        }
        const size_t quarter = n / BYTE_FSE_STREAMS, remainder = n % BYTE_FSE_STREAMS;
        size_t offsets[BYTE_FSE_STREAMS];
        for (size_t i = 0; i < BYTE_FSE_STREAMS; ++i) {
            offsets[i] = i * quarter + (std::min)(i, remainder);
            states[i] = bits[i]->read(table_log);
        }
        auto decode_one = [&](size_t stream, size_t index) {
            const FSE_decode_t entry = table[states[stream]];
            dst[offsets[stream] + index] = entry.symbol;
            states[stream] = entry.newState + bits[stream]->read(entry.nbBits);
        };
        auto decode_fast = [&](size_t stream, size_t index) {
            const FSE_decode_t entry = table[states[stream]];
            dst[offsets[stream] + index] = entry.symbol;
            states[stream] = entry.newState + bits[stream]->read_unchecked(entry.nbBits);
        };
        size_t i = 0;
        for (; quarter - i >= 4; i += 4) {
            const unsigned need = 4 * table_log;
            if (!b0.ensure(need) || !b1.ensure(need) || !b2.ensure(need) || !b3.ensure(need)) break;
            for (unsigned j = 0; j < 4; ++j) {
                decode_fast(0, i + j); decode_fast(1, i + j);
                decode_fast(2, i + j); decode_fast(3, i + j);
            }
        }
        for (; i < quarter; ++i) {
            decode_one(0, i); decode_one(1, i); decode_one(2, i); decode_one(3, i);
        }
        for (size_t j = 0; j < remainder; ++j) decode_one(j, quarter);
        for (size_t j = 0; j < BYTE_FSE_STREAMS; ++j)
            if (states[j] != 0 || !bits[j]->finished())
                throw std::runtime_error("invalid FSE stream ending");
    }

    static size_t encode_split_fse(const T *input, size_t count, uint8_t *&out,
                                   uint8_t *low, uint8_t *high, ByteFseWorkspace &ws) {
        (void)byte_split_bound(count);
        uint8_t *const start = out;
        for (size_t off = 0; off < count;) {
            const size_t len = (std::min)(static_cast<size_t>(FBLK), count - off);
            split_bytes(input + off, len, low, high);
            encode_byte_stream(low, len, out, ws);
            encode_byte_stream(high, len, out, ws);
            off += len;
        }
        return static_cast<size_t>(out - start);
    }

    static void decode_split_fse(const uint8_t *&in, size_t &remaining, size_t count, T *output,
                                 uint8_t *low, uint8_t *high, ByteFseWorkspace &ws) {
        for (size_t off = 0; off < count;) {
            const size_t len = (std::min)(static_cast<size_t>(FBLK), count - off);
            decode_byte_stream(in, remaining, len, low, ws);
            decode_byte_stream(in, remaining, len, high, ws);
            join_bytes(low, high, len, output + off);
            off += len;
        }
    }

    static constexpr size_t RLE_ALIGNMENT = 64;
    struct RleBuffers {
        uint8_t *rcl, *vcl, *rman, *vman;
        size_t vman_capacity;
    };
    struct RleLayout {
        size_t vcl, rman, vman, vman_capacity, bytes;
    };
    static size_t rle_checked_add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a) throw std::bad_array_new_length();
        return a + b;
    }
    static size_t rle_align_up(size_t n) {
        return rle_checked_add(n, RLE_ALIGNMENT - 1) & ~(RLE_ALIGNMENT - 1);
    }
    static size_t rle_vman_capacity(size_t n) {
        // A loaded uint16 mode can differ from an int16 code by up to 98303.
        // Reserve ceil(17*n/8) plus write padding without overflowing 17*n.
        const size_t extra = n / 8 + (n % 8 != 0);
        return rle_checked_add(rle_checked_add(n, n), rle_checked_add(extra, 16));
    }
    static RleLayout rle_layout(size_t n) {
        const size_t vcl = rle_align_up(rle_checked_add(n, 1));
        const size_t rman = rle_checked_add(vcl, vcl);
        const size_t vman = rle_checked_add(rman, rle_align_up(rle_checked_add(n, 16)));
        const size_t vman_capacity = rle_vman_capacity(n);
        const size_t end = rle_checked_add(vman, vman_capacity);
        return {vcl, rman, vman, vman_capacity, rle_checked_add(end, RLE_ALIGNMENT - 1)};
    }
#ifdef __AVX2__
    static constexpr size_t O1T_TILE = 256;
#endif

    struct EncodeWorkspace {
        EncodeBuffer<uint8_t> rle_slab;
        EncodeBuffer<uint8_t> byte_low, byte_high;
        std::unique_ptr<ByteFseWorkspace> byte_fse;
        // Overflow scratch must not reallocate the live RLE slab or tANS streams.
        EncodeBuffer<uint8_t> vcl, vman, ovf_classes, fse_tmp, streams;
        EncodeBuffer<int16_t> ovf;
        EncodeBuffer<FSE_CTable> ct;
        EncodeBuffer<FSE_symbolCompressionTransform> flat_tt;
        EncodeBuffer<uint32_t> histogram_counts;
#ifdef __AVX2__
        uint16_t prepared_indices[O1T_K][O1T_TILE];
#endif
        uint32_t cnt[O1_NCTX * O1T_NSYM];
        uint16_t nf[O1_NCTX * O1T_NSYM];
        uint16_t table_freq[O1_NCTX * O1T_NSYM];
        bool table_valid[O1_NCTX] = {};

        ByteFseWorkspace &byte_fse_workspace() {
            if (!byte_fse) byte_fse.reset(new ByteFseWorkspace);
            return *byte_fse;
        }

        RleBuffers rle_buffers(size_t n) {
            const RleLayout layout = rle_layout(n);
            // Grow the entire slab before deriving any interior pointers.
            void *storage = rle_slab.get(layout.bytes);
            size_t space = layout.bytes;
            auto *base = static_cast<uint8_t *>(
                std::align(RLE_ALIGNMENT, layout.bytes - (RLE_ALIGNMENT - 1), storage, space));
            assert(base != nullptr);
            return {base, base + layout.vcl, base + layout.rman, base + layout.vman,
                    layout.vman_capacity};
        }
    };
    std::unique_ptr<EncodeWorkspace> workspace_;

#ifdef FALCONS_V11_ENTROPY_TEST
    friend struct RleFseEncoderV11TestAccess;
#endif

    EncodeWorkspace &encode_workspace() {
        if (!workspace_) workspace_.reset(new EncodeWorkspace);
        return *workspace_;
    }

    static inline void o1_encode_symbol(BIT_CStream_t &bit, FSE_CState_t &state,
                                       const U16 *state_base,
                                       const FSE_symbolCompressionTransform *flat_tt,
                                       unsigned index) {
        const FSE_symbolCompressionTransform tt = flat_tt[index];
        const U32 nb = static_cast<U32>((state.value + tt.deltaNbBits) >> 16);
        BIT_addBits(&bit, state.value, nb);
        state.value = state_base[(state.value >> nb) + tt.deltaFindState];
    }

#ifdef __AVX2__
    static inline __m256i o1_packed_indices(__m256i previous, __m256i current) {
        const __m256i zero = _mm256_setzero_si256();
        const __m256i four = _mm256_set1_epi16(4);
        // Clamp unsigned first: abs(INT16_MIN) is 0x8000, not a positive signed lane.
        const __m256i a = _mm256_min_epu16(_mm256_abs_epi16(previous), _mm256_set1_epi16(12));
        const __m256i beyond = _mm256_add_epi16(_mm256_cmpgt_epi16(a, four),
            _mm256_add_epi16(_mm256_cmpgt_epi16(a, _mm256_set1_epi16(7)),
                             _mm256_cmpgt_epi16(a, _mm256_set1_epi16(11))));
        const __m256i bucket = _mm256_sub_epi16(_mm256_min_epu16(a, four), beyond);
        __m256i context = _mm256_add_epi16(
            _mm256_sub_epi16(_mm256_slli_epi16(bucket, 1), _mm256_set1_epi16(1)),
            _mm256_srli_epi16(previous, 15));
        context = _mm256_andnot_si256(_mm256_cmpeq_epi16(a, zero), context);
        const __m256i zigzag = _mm256_xor_si256(_mm256_slli_epi16(current, 1),
                                              _mm256_srai_epi16(current, 15));
        const __m256i symbol = _mm256_min_epu16(zigzag, _mm256_set1_epi16(255));
        return _mm256_or_si256(_mm256_slli_epi16(context, 8), symbol);
    }

    static inline void o1_prepare_indices(const T *current, size_t count, uint16_t *indices,
                                          const O1Lut &L) {
        size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            const __m256i previous = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(current + i - 1));
            const __m256i code = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(current + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(indices + i), o1_packed_indices(previous, code));
        }
        for (; i < count; ++i)
            indices[i] = O1T_NSYM * L.ctx[static_cast<uint16_t>((current + i)[-1])] +
                         L.sym256[static_cast<uint16_t>(current[i])];
    }

    // Keep transition state in registers, separate from histogram and tile preparation.
    FORCE_NOINLINE void o1_encode_tile(BIT_CStream_t *bc, FSE_CState_t *cst,
                                      const U16 *state_table,
                                      const FSE_symbolCompressionTransform *transforms,
                                      const uint16_t (*indices)[O1T_TILE], size_t jj) {
        assert(jj > 1);
        const size_t count = std::min(O1T_TILE, jj - 1);
        const unsigned phase = static_cast<unsigned>(jj - count) & 3;
        U32 s0 = static_cast<U32>(cst[0].value), s1 = static_cast<U32>(cst[1].value);
        U32 s2 = static_cast<U32>(cst[2].value), s3 = static_cast<U32>(cst[3].value);
        size_t b0 = bc[0].bitContainer, b1 = bc[1].bitContainer;
        size_t b2 = bc[2].bitContainer, b3 = bc[3].bitContainer;
        unsigned p0 = bc[0].bitPos, p1 = bc[1].bitPos, p2 = bc[2].bitPos, p3 = bc[3].bitPos;
        char *o0 = bc[0].ptr, *o1 = bc[1].ptr, *o2 = bc[2].ptr, *o3 = bc[3].ptr;
        const auto step = [=](U32 &state, size_t &bits, unsigned &pos, unsigned index) {
            const FSE_symbolCompressionTransform tt = transforms[index];
            assert(state >= O1T_M && state < 2 * O1T_M);
            const U32 nb = (state + tt.deltaNbBits) >> 16;
            assert(nb <= O1T_PB && nb + pos < sizeof(bits) * 8);
            bits |= (static_cast<size_t>(state) & BIT_mask[nb]) << pos;
            pos += nb;
            state = state_table[static_cast<ptrdiff_t>(state >> nb) + tt.deltaFindState];
        };
        const auto flush = [](size_t &bits, unsigned &pos, char *&out, char *end) {
            const unsigned bytes = pos >> 3;
            assert(pos < sizeof(bits) * 8 && out <= end);
            MEM_writeLEST(out, bits);
            out += bytes;
            if (out > end) out = end;
            bits >>= bytes * 8;
            pos &= 7;
        };
        for (size_t t = count; t > 0;) {
            --t;
            step(s0, b0, p0, indices[0][t]);
            step(s1, b1, p1, indices[1][t]);
            step(s2, b2, p2, indices[2][t]);
            step(s3, b3, p3, indices[3][t]);
            if (((t + phase) & 3) == 0) {
                flush(b0, p0, o0, bc[0].endPtr);
                flush(b1, p1, o1, bc[1].endPtr);
                flush(b2, p2, o2, bc[2].endPtr);
                flush(b3, p3, o3, bc[3].endPtr);
            }
        }
        cst[0].value = s0; cst[1].value = s1; cst[2].value = s2; cst[3].value = s3;
        bc[0].bitContainer = b0; bc[0].bitPos = p0; bc[0].ptr = o0;
        bc[1].bitContainer = b1; bc[1].bitPos = p1; bc[1].ptr = o1;
        bc[2].bitContainer = b2; bc[2].bitPos = p2; bc[2].ptr = o2;
        bc[3].bitContainer = b3; bc[3].bitPos = p3; bc[3].ptr = o3;
    }
#endif

    // ---- Generic K-stream context-switching tANS (method_==1) ----
    // K independent FSE bitstreams (chunks), each switching (stateTable,symbolTT)/DTable per symbol
    // by the 15-way context. Interleaved -> K independent chains hide the FSE state latency (ILP).
    size_t encode_o1_ovf(const int16_t *ovf, size_t n, uchar *&bytes) {
        uchar *sb = bytes;
        if (n == 0) {
            write(static_cast<uint8_t>(0), bytes);
            return bytes - sb;
        }
        // Every escape has at least one mantissa byte, so the headers alone
        // make a compressed candidate no smaller than raw for n <= 16.
        if (n <= 2 * sizeof(uint64_t)) {
            write(static_cast<uint8_t>(0), bytes);
            memcpy(bytes, ovf, n * sizeof(int16_t));
            bytes += n * sizeof(int16_t);
            return bytes - sb;
        }

        EncodeWorkspace &ws = encode_workspace();
        uint8_t *vcl = ws.vcl.get(n);
        // These are signed int16 escapes relative to zero, needing at most 16 mantissa bits.
        uint8_t *vman = ws.vman.get(2 * n + 16);
        size_t vbp = 0;
        uint64_t vacc = 0;
        int van = 0;
        for (size_t i = 0; i < n; i++) {
            int32_t fct = ovf[i];
            uint32_t zz = (static_cast<uint32_t>(fct) << 1) ^ static_cast<uint32_t>(fct >> 31);
            uint64_t vv = static_cast<uint64_t>(zz) + 1;
            int vk = 63 - __builtin_clzll(vv);
            vcl[i] = static_cast<uint8_t>(vk);
            vacc = (vacc << vk) | (vv & ((1ull << vk) - 1));
            van += vk;
            // Escapes have 8 <= vk <= 16, with at most seven carry bits.
            assert(vk >= 8 && vk <= 16);
            van -= 8;
            vman[vbp++] = static_cast<uint8_t>(vacc >> van);
            if (van >= 8) {
                van -= 8;
                vman[vbp++] = static_cast<uint8_t>(vacc >> van);
            }
        }
        if (van > 0)
            vman[vbp++] = static_cast<uint8_t>(vacc << (8 - van));

        uint8_t *o_vcl = ws.ovf_classes.get(n + n / FBLK * 8 + 64);
        size_t s_vcl = fse_enc(vcl, n, o_vcl);
        const size_t raw_size = n * sizeof(int16_t);
        const size_t cm_size = s_vcl + vbp + 2 * sizeof(uint64_t);
        if (cm_size >= raw_size) {
            write(static_cast<uint8_t>(0), bytes);
            memcpy(bytes, ovf, n * sizeof(int16_t));
            bytes += n * sizeof(int16_t);
        } else {
            write(static_cast<uint8_t>(1), bytes);
            write(static_cast<uint64_t>(s_vcl), bytes);
            write(static_cast<uint64_t>(vbp), bytes);
            memcpy(bytes, o_vcl, s_vcl);
            bytes += s_vcl;
            memcpy(bytes, vman, vbp);
            bytes += vbp;
        }
        return bytes - sb;
    }

    static int16_t *decode_o1_ovf(const uchar *&bytes, size_t n) {
        uint8_t mode;
        read(mode, bytes);
        int16_t *ovf = static_cast<int16_t *>(malloc((n ? n : 1) * sizeof(int16_t)));
        if (n == 0)
            return ovf;

        if (mode == 0) {
            memcpy(ovf, bytes, n * sizeof(int16_t));
            bytes += n * sizeof(int16_t);
            return ovf;
        }

        uint64_t s_vcl, vbp;
        read(s_vcl, bytes);
        read(vbp, bytes);

        const uchar *o_vcl = bytes;
        bytes += s_vcl;
        const uchar *vman = bytes;
        bytes += vbp;

        uint8_t *vcl = static_cast<uint8_t *>(malloc(n));
        fse_dec(o_vcl, n, vcl);
        size_t vrp = 0;
        uint64_t vda = 0;
        int vdn = 0;
        for (size_t i = 0; i < n; i++) {
            int vk = vcl[i];
            while (vdn < vk) {
                vda = (vda << 8) | vman[vrp++];
                vdn += 8;
            }
            vdn -= vk;
            uint64_t vv = (1ull << vk) | ((vda >> vdn) & ((1ull << vk) - 1));
            uint32_t zz = static_cast<uint32_t>(vv - 1);
            int32_t fct = static_cast<int32_t>((zz >> 1) ^ (~(zz & 1) + 1));
            ovf[i] = static_cast<int16_t>(fct);
        }
        free(vcl);
        return ovf;
    }

    static constexpr size_t O1T_SEG = static_cast<size_t>(8) << 20;   // target symbols per table segment

    // Quant statistics drift along the stream (levels, passes, level-wise eb), so one global
    // table set is a compromise: per-segment tables measured ~2-3% tighter payload at identical
    // per-symbol cost. Each segment is a self-contained block (own tables, K streams, escapes).
    size_t encode_h1fse(const T *bins, size_t N, uchar *&bytes) {
        uchar *sb = bytes;
        size_t S = N / O1T_SEG;
        if (S < 1) S = 1;
        if (S > 16) S = 16;
        write(static_cast<uint64_t>(S), bytes);
        size_t seg = (N + S - 1) / S;
        for (size_t g = 0; g < S; g++) {
            size_t lo = g * seg, hi = lo + seg;
            if (hi > N) hi = N;
            encode_h1fse_block(bins + lo, hi - lo, bytes);
        }
        return static_cast<size_t>(bytes - sb);
    }

    size_t encode_h1fse_block(const T *bins, size_t N, uchar *&bytes) {
        const O1Lut &L = o1_lut();
        uchar *sb = bytes;
        // if (N == 0) { 
        //     write((uint64_t)0, bytes); 
        //     write((uint64_t)O1T_K, bytes); 
        //     return (size_t)(bytes - sb); 
        // }
        size_t cs = (N + O1T_K - 1) / O1T_K;
        EncodeWorkspace &ws = encode_workspace();
        int16_t *ovf = ws.ovf.get(N);
        size_t novf = 0;
        size_t novf_k[16] = {0};
        uint32_t *cnt = ws.cnt;
        memset(cnt, 0, sizeof(ws.cnt));
        //std::getenv("FALCONS_TANS_FULLHIST") || 
        if (N < (1u << 20)) {       // exact full per-context histogram (opt-out via env / small N); default = fast sampled path below
            const size_t total = O1_NCTX * O1T_NSYM;
            uint32_t *cnt1 = nullptr, *cnt2 = nullptr, *cnt3 = nullptr;
            if (N >= (1u << 15)) {
                cnt1 = ws.histogram_counts.get(3 * total);
                cnt2 = cnt1 + total;
                cnt3 = cnt2 + total;
                memset(cnt1, 0, 3 * total * sizeof(*cnt1));
            }
            for (int k = 0; k < O1T_K; k++) { 
                size_t lo = cs * k, hi = lo + cs; 
                if (hi > N) 
                    hi = N;
                uint16_t prev = 0;
                size_t i = lo;
                if (cnt1) {
                    for (; i + 4 <= hi; i += 4) {
                        const uint16_t c0 = static_cast<uint16_t>(bins[i]);
                        const uint16_t c1 = static_cast<uint16_t>(bins[i + 1]);
                        const uint16_t c2 = static_cast<uint16_t>(bins[i + 2]);
                        const uint16_t c3 = static_cast<uint16_t>(bins[i + 3]);
                        const unsigned s0 = L.sym256[c0], s1 = L.sym256[c1];
                        const unsigned s2 = L.sym256[c2], s3 = L.sym256[c3];
                        cnt[O1T_NSYM * L.ctx[prev] + s0]++;
                        cnt1[O1T_NSYM * L.ctx[c0] + s1]++;
                        cnt2[O1T_NSYM * L.ctx[c1] + s2]++;
                        cnt3[O1T_NSYM * L.ctx[c2] + s3]++;
                        if (s0 == O1T_NSYM - 1) { ovf[novf++] = c0; novf_k[k]++; }
                        if (s1 == O1T_NSYM - 1) { ovf[novf++] = c1; novf_k[k]++; }
                        if (s2 == O1T_NSYM - 1) { ovf[novf++] = c2; novf_k[k]++; }
                        if (s3 == O1T_NSYM - 1) { ovf[novf++] = c3; novf_k[k]++; }
                        prev = c3;
                    }
                }
                for (; i < hi; i++) {
                    uint16_t code = bins[i];
                    int c = L.ctx[prev];
                    int s = L.sym256[code];
                    if (s == O1T_NSYM - 1) { 
                        ovf[novf++] = code; 
                        novf_k[k]++; 
                    }
                    cnt[static_cast<size_t>(c) * O1T_NSYM + s]++; 
                    prev = code; 
                } 
            }
            if (cnt1)
                for (size_t t = 0; t < total; ++t)
                    cnt[t] += cnt1[t] + cnt2[t] + cnt3[t];
        } else {                                                        // fast: full escape scan + windowed-sample histogram
#ifdef __AVX2__
            const __m256i min_code = _mm256_set1_epi16(-127);
            const __m256i max_code = _mm256_set1_epi16(127);
#endif
            for (int k = 0; k < O1T_K; k++) { 
                size_t lo = cs * k, hi = lo + cs; 
                if (hi > N) 
                    hi = N; 
                size_t b0 = novf;
                size_t i = lo;
#ifdef __AVX2__
                // Escape is exactly outside [-127, 127]; process each stream in order.
                for (; i + 16 <= hi; i += 16) {
                    const __m256i values = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(bins + i));
                    const __m256i escapes = _mm256_or_si256(
                        _mm256_cmpgt_epi16(min_code, values), _mm256_cmpgt_epi16(values, max_code));
                    uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(escapes));
                    if (mask == 0) continue;
                    if (mask == 0xFFFFFFFFu) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ovf + novf), values);
                        novf += 16;
                    } else {
                        mask &= 0x55555555u;
                        do {
                            const unsigned lane = static_cast<unsigned>(__builtin_ctz(mask)) >> 1;
                            ovf[novf++] = bins[i + lane];
                            mask &= mask - 1;
                        } while (mask);
                    }
                }
#endif
                for (; i < hi; i++) {
                    ovf[novf] = bins[i];
                    novf += (L.sym256[static_cast<uint16_t>(bins[i])] == O1T_NSYM - 1);
                }
                novf_k[k] = novf - b0; 
            }
            const size_t total = O1_NCTX * O1T_NSYM;
            uint32_t *cnt1 = ws.histogram_counts.get(3 * total);
            uint32_t *cnt2 = cnt1 + total, *cnt3 = cnt2 + total;
            memset(cnt1, 0, 3 * total * sizeof(*cnt1));
            size_t nwin = 2048, wlen = 512;                             // ~1M windowed pairs per block; prev exact within each window
            size_t step = N / nwin; 
            if (step < wlen) 
                step = wlen;
            for (size_t w = 0; w < N; w += step) { 
                int prev = (w > 0) ? bins[w - 1] : 0; 
                size_t we = w + wlen; 
                if (we > N) 
                    we = N;
                size_t i = w;
                for (; i + 4 <= we; i += 4) {
                    const uint16_t c0 = static_cast<uint16_t>(bins[i]);
                    const uint16_t c1 = static_cast<uint16_t>(bins[i + 1]);
                    const uint16_t c2 = static_cast<uint16_t>(bins[i + 2]);
                    const uint16_t c3 = static_cast<uint16_t>(bins[i + 3]);
                    cnt[O1T_NSYM * L.ctx[static_cast<uint16_t>(prev)] + L.sym256[c0]]++;
                    cnt1[O1T_NSYM * L.ctx[c0] + L.sym256[c1]]++;
                    cnt2[O1T_NSYM * L.ctx[c1] + L.sym256[c2]]++;
                    cnt3[O1T_NSYM * L.ctx[c2] + L.sym256[c3]]++;
                    prev = bins[i + 3];
                }
                for (; i < we; i++) {
                    int code = bins[i];
                    cnt[static_cast<size_t>(O1T_NSYM) * L.ctx[static_cast<uint16_t>(prev)] + L.sym256[static_cast<uint16_t>(code)]]++;
                    prev = code; 
                } 
            }
            for (size_t t = 0; t < total; t++) {
                cnt[t] += cnt1[t] + cnt2[t] + cnt3[t];
                cnt[t]++;   // Laplace +1: no zero-freq -> any real symbol stays encodable
            }
        }
        
        uint16_t *nf = ws.nf;

        const U16 *stTab[O1_NCTX];
        const FSE_symbolCompressionTransform *syTT[O1_NCTX];
        size_t ctsz = FSE_CTABLE_SIZE_U32(O1T_PB, O1T_NSYM - 1);
        FSE_CTable *ct[O1_NCTX];
        FSE_CTable *ct_storage = ws.ct.get(O1_NCTX * ctsz);

        int16_t normc[O1T_NSYM];
        for (int c = 0; c < O1_NCTX; c++) {
            o1_normalize(cnt + c * O1T_NSYM, nf + c * O1T_NSYM, O1T_NSYM, O1T_M);
            ct[c] = ct_storage + static_cast<size_t>(c) * ctsz;
            const uint16_t *freq = nf + c * O1T_NSYM;
            uint16_t *cached = ws.table_freq + c * O1T_NSYM;
            if (!ws.table_valid[c] || memcmp(freq, cached, O1T_NSYM * sizeof(*freq)) != 0) {
                int same = 0;
                for (; same < c; same++)
                    if (memcmp(freq, nf + same * O1T_NSYM, O1T_NSYM * sizeof(*freq)) == 0)
                        break;
                if (same < c) {
                    ct[c] = ct[same];
                } else {
                    for (int s = 0; s < O1T_NSYM; s++)
                        normc[s] = static_cast<int16_t>(freq[s]);
                    FSE_buildCTable(ct[c], normc, O1T_NSYM - 1, O1T_PB);
                    memcpy(cached, freq, O1T_NSYM * sizeof(*freq));
                    ws.table_valid[c] = true;
                }
            }
            FSE_CState_t tmp; 
            FSE_initCState(&tmp, ct[c]); 
            stTab[c] = static_cast<const U16 *>(tmp.stateTable);
            syTT[c] = static_cast<const FSE_symbolCompressionTransform *>(tmp.symbolTT);
        }
        static_assert(sizeof(FSE_CTable) == 2 * sizeof(U16), "Vendored FSE CT word layout changed");
        static_assert(2 * O1_NCTX * FSE_CTABLE_SIZE_U32(O1T_PB, O1T_NSYM - 1) + O1T_M <=
                          static_cast<unsigned>(std::numeric_limits<int>::max()),
                      "Flat CT transform offsets must fit int");
        const U16 *const encode_st = static_cast<const U16 *>(static_cast<const void *>(ct_storage));
        FSE_symbolCompressionTransform *const encode_tt = ws.flat_tt.get(O1_NCTX * O1T_NSYM);
        for (int c = 0; c < O1_NCTX; ++c) {
            // Rebase within the CT allocation, including FSE's two-U16 header.
            const int offset = static_cast<int>(2 * (ct[c] - ct_storage) + 2);
            assert(stTab[c] == encode_st + offset);
            assert(static_cast<size_t>(offset) + O1T_M <= 2 * O1_NCTX * ctsz);
            for (int s = 0; s < O1T_NSYM; ++s) {
                auto &tt = encode_tt[c * O1T_NSYM + s];
                tt.deltaNbBits = syTT[c][s].deltaNbBits;
                // Absent symbols leave deltaFindState uninitialized in vendored FSE.
                tt.deltaFindState = nf[c * O1T_NSYM + s] ? syTT[c][s].deltaFindState + offset : 0;
            }
        }
        uint8_t *buf[16]; 
        BIT_CStream_t bc[16]; 
        FSE_CState_t cst[16]; 
        size_t initIdx[16]; 
        size_t capk = cs * 2 + 4096;
        uint8_t *streams = ws.streams.get(O1T_K * capk);
        for (int k = 0; k < O1T_K; k++) {
            buf[k] = streams + k * capk;
            BIT_initCStream(&bc[k], buf[k], capk);
            size_t lo = k * cs, hi = lo + cs; 
            if (hi > N) 
                hi = N;
            if (hi > lo) { 
                size_t idx = hi - 1; 
                initIdx[k] = idx;          // init each stream with its last symbol (reverse)
                int prevc = (idx > lo) ? bins[idx - 1] : 0; 
                int c = L.ctx[static_cast<uint16_t>(prevc)]; int s = L.sym256[static_cast<uint16_t>(bins[idx])];
                FSE_initCState2(&cst[k], ct[c], static_cast<int16_t>(s)); 
            }
            else 
                initIdx[k] = -1;
        }
        // Peel unequal tails and last-symbol initialization before the four-stream interior.
        size_t jj = cs;
        const size_t interior = N >= 4 * O1T_K ? N - (O1T_K - 1) * cs - 1 : 0;
        while (jj > interior) {
            --jj;
            for (int k = 0; k < O1T_K; k++) { 
                size_t idx = k * cs + jj; 
                if (idx >= N || idx == initIdx[k]) 
                    continue;
                size_t lo = k * cs; 
                int prevc = (idx > lo) ? bins[idx - 1] : 0;
                int c = L.ctx[static_cast<uint16_t>(prevc)]; 
                int s = L.sym256[static_cast<uint16_t>(bins[idx])];
                o1_encode_symbol(bc[k], cst[k], encode_st, encode_tt, c * O1T_NSYM + s);
            }
            if ((jj & 3) == 0) 
                for (int k = 0; k < O1T_K; k++) 
                    BIT_flushBits(&bc[k]); 
        }
        if (interior > 0) {
#ifdef __AVX2__
            // Prepare forward in bounded tiles; consume backward at the original flush points.
            while (jj > 1) {
                const size_t count = std::min(O1T_TILE, jj - 1);
                const size_t begin = jj - count;
                for (int k = 0; k < O1T_K; ++k)
                    o1_prepare_indices(bins + k * cs + begin, count, ws.prepared_indices[k], L);
                o1_encode_tile(bc, cst, encode_st, encode_tt, ws.prepared_indices, jj);
                jj = begin;
            }
            const uint16_t s0 = bins[0], s1 = bins[cs], s2 = bins[2 * cs], s3 = bins[3 * cs];
#else
            const T *p0 = bins + jj - 1;
            const T *p1 = p0 + cs, *p2 = p1 + cs, *p3 = p2 + cs;
            uint16_t s0 = *p0, s1 = *p1, s2 = *p2, s3 = *p3;
            while (jj > 1) {
                --jj;
                const uint16_t prev0 = p0[-1], prev1 = p1[-1], prev2 = p2[-1], prev3 = p3[-1];
                o1_encode_symbol(bc[0], cst[0], encode_st, encode_tt, O1T_NSYM * L.ctx[prev0] + L.sym256[s0]);
                o1_encode_symbol(bc[1], cst[1], encode_st, encode_tt, O1T_NSYM * L.ctx[prev1] + L.sym256[s1]);
                o1_encode_symbol(bc[2], cst[2], encode_st, encode_tt, O1T_NSYM * L.ctx[prev2] + L.sym256[s2]);
                o1_encode_symbol(bc[3], cst[3], encode_st, encode_tt, O1T_NSYM * L.ctx[prev3] + L.sym256[s3]);
                s0 = prev0; s1 = prev1; s2 = prev2; s3 = prev3;
                --p0; --p1; --p2; --p3;
                if ((jj & 3) == 0)
                    for (int k = 0; k < O1T_K; ++k)
                        BIT_flushBits(&bc[k]);
            }
#endif
            o1_encode_symbol(bc[0], cst[0], encode_st, encode_tt, L.sym256[s0]);
            o1_encode_symbol(bc[1], cst[1], encode_st, encode_tt, L.sym256[s1]);
            o1_encode_symbol(bc[2], cst[2], encode_st, encode_tt, L.sym256[s2]);
            o1_encode_symbol(bc[3], cst[3], encode_st, encode_tt, L.sym256[s3]);
            for (int k = 0; k < O1T_K; ++k)
                BIT_flushBits(&bc[k]);
        }  // Four transitions plus carry fit the existing 64-bit FSE bit container.
        size_t blen[16];
        for (int k = 0; k < O1T_K; k++) { 
            if (initIdx[k] != static_cast<size_t>(-1)) { 
                FSE_flushCState(&bc[k], &cst[k]); 
                blen[k] = BIT_closeCStream(&bc[k]); 
            } 
            else
                blen[k] = 0; 
        }
        write(static_cast<uint64_t>(N), bytes); 
        write(static_cast<uint64_t>(O1T_K), bytes);
        for (int k = 0; k < O1T_K; k++) 
            write(static_cast<uint64_t>(blen[k]), bytes);
        for (int k = 0; k < O1T_K; k++) 
            write(static_cast<uint64_t>(novf_k[k]), bytes);
        memcpy(bytes, nf, O1_NCTX * O1T_NSYM * 2); 
        bytes += O1_NCTX * O1T_NSYM * 2;
        for (int k = 0; k < O1T_K; k++) { 
            memcpy(bytes, buf[k], blen[k]); bytes += blen[k]; 
        }
        encode_o1_ovf(ovf, novf, bytes);
        // if (std::getenv("FALCONS_O1DBG")) { 
        //     size_t bt = 0; 
        //     for (int k = 0; k < O1T_K; k++) 
        //     bt += blen[k];
        //     std::fprintf(stderr, "[O1tans] N=%zu O1T_K=%d tANS=%zu(%.3f b/sym) novf=%zu(%.1f%%)\n", N, O1T_K, bt, 
        //         bt * 8.0 / N, novf, 100.0 * novf / N); 
        // }
        return (bytes - sb);
    }

    T *decode_h1fse(const uchar *&bytes, size_t N) {
        constexpr size_t pad = 64 / sizeof(T); 
        T *out = new T[N + pad];
#if defined(__linux__)
        {   // hint THP for the decoded quant array (large hot buffer; big decompress win)
            uintptr_t _mb = reinterpret_cast<uintptr_t>(out) & ~static_cast<uintptr_t>(4095);
            madvise(reinterpret_cast<void *>(_mb), (N + pad) * sizeof(T) + (reinterpret_cast<uintptr_t>(out) - _mb), MADV_HUGEPAGE);
        }
#endif
        uint64_t S_;
        read(S_, bytes);
        size_t S = S_ ? static_cast<size_t>(S_) : 1;
        size_t seg = (N + S - 1) / S;
        for (size_t g = 0; g < S; g++) {
            size_t lo = g * seg, hi = lo + seg;
            if (hi > N) hi = N;
            decode_h1fse_block(bytes, hi - lo, out + lo);
        }
        return out;
    }

    void decode_h1fse_block(const uchar *&bytes, size_t N, T *out) {
        uint64_t n_, Kk;
        read(n_, bytes);
        read(Kk, bytes);
        int K = static_cast<int>(Kk);
        if (N == 0)
            return;
        size_t cs = (N + K - 1) / K;
        uint64_t blen[16], nvk[16]; 
        for (int k = 0; k < K; k++) 
            read(blen[k], bytes); 
        for (int k = 0; k < K; k++) 
            read(nvk[k], bytes);
        uint16_t *nf = static_cast<uint16_t *>(malloc(O1_NCTX * O1T_NSYM * 2));
        memcpy(nf, bytes, O1_NCTX * O1T_NSYM * 2); 
        bytes += O1_NCTX * O1T_NSYM * 2;
        const uint8_t *sbase = static_cast<const uint8_t *>(bytes); 
        size_t totbl = 0; 
        for (int k = 0; k < K; k++) 
            totbl += blen[k]; 
        bytes += totbl;
        size_t totov = 0; 

        for (int k = 0; k < K; k++) 
            totov += nvk[k]; 
        int16_t *ovf = decode_o1_ovf(bytes, totov);
        size_t dtsz = FSE_DTABLE_SIZE_U32(O1T_PB);
        FSE_DTable *dt[O1_NCTX];
        FSE_DTable *dt_storage = static_cast<FSE_DTable *>(malloc(O1_NCTX * dtsz * sizeof(FSE_DTable)));
        int16_t normc[O1T_NSYM];
        for (int c = 0; c < O1_NCTX; c++) {
            for (int s = 0; s < O1T_NSYM; s++) 
                normc[s] = static_cast<int16_t>(nf[c * O1T_NSYM + s]);

            dt[c] = dt_storage + static_cast<size_t>(c) * dtsz;
            FSE_buildDTable(dt[c], normc, O1T_NSYM - 1, O1T_PB);
        }
        const O1Lut &L = o1_lut();
        BIT_DStream_t bd[16]; 
        FSE_DState_t ds[16]; 
        size_t ovi[16]; int prev[16];
        size_t off = 0, ooff = 0;
        for (int k = 0; k < K; k++) { 
            const uint8_t *s = sbase + off; 
            off += blen[k];
            if (blen[k] > 0) { 
                BIT_initDStream(&bd[k], s, blen[k]); 
                FSE_initDState(&ds[k], &bd[k], dt[L.ctx[0]]);
            }
            ovi[k] = ooff; ooff += nvk[k]; prev[k] = 0; 
        }
        for (size_t jj = 0; jj < cs; jj++) {                              // interleaved forward; K independent chains -> ILP
            if ((jj & 3) == 0) for (int k = 0; k < K; k++) { 
                if (blen[k]) 
                    BIT_reloadDStream(&bd[k]); 
            }  // reload every 4 (4*11<57)
            for (int k = 0; k < K; k++) { 
                size_t idx = k * cs + jj; 
                if (idx >= N) 
                    continue;
                int c = L.ctx[static_cast<uint16_t>(prev[k])]; 
                ds[k].table = dt[c] + 1;
                unsigned s = FSE_decodeSymbol(&ds[k], &bd[k]);
                int code = (s == O1T_NSYM - 1) ? ovf[ovi[k]++] : o1_unzz(s);
                out[idx] = static_cast<T>(code); 
                prev[k] = code; 
            } 
        }
        free(dt_storage); free(nf); free(ovf);
    }

   public:
    RleFseEncoder() = default;
    // Model flags copy, but mutable scratch and cached tables remain object-private.
    RleFseEncoder(const RleFseEncoder &other)
        : lo_(other.lo_), mode_(other.mode_), method_(other.method_),
          use_delta_(other.use_delta_), scan_simd_(other.scan_simd_),
          decode_remaining_(other.decode_remaining_) {}
    RleFseEncoder &operator=(const RleFseEncoder &other) {
        if (this != &other) {
            lo_ = other.lo_; mode_ = other.mode_; method_ = other.method_;
            use_delta_ = other.use_delta_; scan_simd_ = other.scan_simd_;
            decode_remaining_ = other.decode_remaining_;
            workspace_.reset();
        }
        return *this;
    }
    RleFseEncoder(RleFseEncoder &&) = default;
    RleFseEncoder &operator=(RleFseEncoder &&) = default;

    // void preprocess_encode(const T *bins, size_t num_bin, int stateNum, size_t *frequencyList) {
    //     (void)frequencyList;               // no histogram needed (mode = radius, known a priori)
    //     derive(bins, num_bin, stateNum);
    // }
    void preprocess_encode(const T *bins, size_t num_bin, int stateNum) {
        // static const bool fse_ok = o1_fse_selftest();   // link self-test (tableLog vs linked FSE); uncomment to re-arm
        // (void)fse_ok;
        (void)stateNum;
        mode_ = 0; lo_ = 0;
        // ONE windowed sample (256 windows x 32 contiguous) -> two signals:
        //   p0     = fraction(code==mode)        : long-run-ness (RLE friendliness)
        //   dzeros = fraction(code==prev) - p0   : extra zeros delta would create (delta benefit)
        // delta if dzeros>0.005; RLE if p0>=0.80 else o1tans (delta orthogonal); SIMD scan if p0>=0.85.
        const T mode = static_cast<T>(mode_);
        size_t NW = 256, WL = 32;
        if (num_bin < NW * WL) { 
            WL = num_bin / NW; 
            if (!WL) 
                WL = 1; 
        }
        size_t step = (num_bin > NW * WL) ? (num_bin - WL) / NW : WL; 
        if (!step) step = 1;
        size_t z = 0, zd = 0, cnt = 0, cntd = 0;
        size_t raw_escapes = 0;
        for (size_t w = 0; w < NW; w++) {
            size_t off = w * step; 
            if (off + WL > num_bin) 
                break;
            T prev = bins[off];
            for (size_t t = 0; t < WL; t++) {
                T x = bins[off + t];
                const uint16_t raw_bits = static_cast<uint16_t>(x);
                const int raw = raw_bits < 32768 ? int(raw_bits) : int(raw_bits) - 65536;
                raw_escapes += raw < -127 || raw > 127;
                if (x == mode) 
                    z++;
                if (t > 0 && x == prev) 
                    zd++;
                prev = x;
            }
            cnt += WL; 
            cntd += (WL - 1);
        }
        double p0 = cnt ? 1.0 * z / cnt : 1.0;
        double dzeros = (cnt && cntd) ? (1.0 * zd / cntd - p0) : 0.0;
        use_delta_ = (dzeros > 0.005);
        method_ = (p0 >= 0.80) ? 0 : 1;   // RLE(long runs) vs o1tans(tight); delta orthogonal -> delta+RLE / delta+o1tans
        // Order-1 tANS carries a fixed O1_NCTX*O1T_NSYM*2 (~15.9 KB) context table. It only wins
        // once that table is a small fraction (<=~10%) of the stream; for smaller inputs the
        // table-free RLE+FSE path (method 0) is both smaller AND can't overflow the compress
        // buffer (which is sized from the input), so fall back to it.
        if (num_bin < static_cast<size_t>(20) * O1_NCTX * O1T_NSYM) method_ = 0;
        const double scan_p0 = use_delta_ && cntd ? 1.0 * zd / cntd : p0;
        scan_simd_ = (scan_p0 >= 0.85);
        if (method_ == 1 && cnt && raw_escapes >= cnt - cnt / 2)
            method_ = 2;
    }

    size_t encode(const T *bins0, size_t N, uchar *&bytes) {
        const T *bins = bins0;
        // forward-diff IN PLACE when use_delta_: iterate BACKWARD so a[i-1] is still original (no extra buffer).
        // const_cast is safe: caller (interp) consumes the quant array right after encode (reallocated each
        // compress, never read again). a[0] is kept as the seed; un-delta on decode reverses it (stream flag).
        if (use_delta_ && N > 0) {
            T *a = const_cast<T *>(bins0);
            size_t i = N;
#ifdef __AVX2__
            for (; i >= 17; i -= 16)   // block [i-16,i-1] -= [i-17,i-2]; both loads before store, lower block still original
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(a + i - 16),
                    _mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i - 16)),
                                     _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i - 17))));
#elif defined(__ARM_FEATURE_SVE2)
            const size_t lanes = svcnth();
            const svbool_t pg = svptrue_b16();
            for (; i >= lanes + 1; i -= lanes) {
                uint16_t *current = reinterpret_cast<uint16_t *>(a + i - lanes);
                const uint16_t *previous = reinterpret_cast<const uint16_t *>(a + i - lanes - 1);
                const svuint16_t delta = svsub_u16_x(
                    pg, svld1_u16(pg, current), svld1_u16(pg, previous));
                svst1_u16(pg, current, delta);
            }
#endif
            for (; i > 1; i--) a[i - 1] = static_cast<T>(a[i - 1] - a[i - 2]);   // a[0] stays as seed
            bins = a;
        }
        if (method_ == 2) {
            EncodeWorkspace &ws = encode_workspace();
            return encode_split_fse(bins, N, bytes, ws.byte_low.get(FBLK),
                                    ws.byte_high.get(FBLK), ws.byte_fse_workspace());
        }
        if (method_ == 1) return encode_h1fse(bins, N, bytes);   // 1 = order1-tANS
        // if (method_ == 2) return encode_rle_adt(bins, N, bytes);  // 2 = RLE+ADT2+zstd
        // if (method_ == 3) return encode_adt(bins, N, bytes);      // 3 = ADT2+zstd (noRLE)
        // method_ == 0 -> RLE+FSE+zstd (fallthrough)
        const T mode = static_cast<T>(mode_);
        const int32_t modev = static_cast<int32_t>(mode_);
        EncodeWorkspace &ws = encode_workspace();
        const RleBuffers scratch = ws.rle_buffers(N);
        uint8_t *rcl = scratch.rcl, *vcl = scratch.vcl;
        uint8_t *rman = scratch.rman, *vman = scratch.vman;
        size_t nr = 0, nv = 0, rbp = 0, vbp = 0; 
        uint64_t racc = 0, vacc = 0; 
        int ran = 0, van = 0;
        const size_t vman_capacity = scratch.vman_capacity;
#ifdef __AVX2__
        const __m256i mode_vector = _mm256_set1_epi16(static_cast<int16_t>(mode));
        size_t block_end = 0;
        uint32_t nonmode_mask = 0;
#endif
        size_t i = 0;
        while (i < N) {
            size_t L;
#ifdef __AVX2__
            if (scan_simd_)
                L = rle_scan_run_cached(bins, i, N, mode, mode_vector, block_end, nonmode_mask);
            else
#endif
                L = scan_run(bins, i, N, mode, scan_simd_);
            i += L;
            uint64_t v = L + 1; 
            int k = 63 - __builtin_clzll(v); 
            rcl[nr++] = k;
            if (k) {
                racc = (racc << k) | (v & ((1ull << k) - 1));
                ran += k;
                while (ran >= 8) {
                    ran -= 8;
                    rman[rbp++] = (racc >> ran);
                }
            }

            if (i >= N) break;

            int32_t fct = bins[i++] - modev; 
            uint32_t zz = (static_cast<uint32_t>(fct) << 1) ^ static_cast<uint32_t>(fct >> 31);
            uint64_t vv = static_cast<uint32_t>(zz) + 1; 
            int vk = 63 - __builtin_clzll(vv); 
            vcl[nv++] = vk;
            vacc = (vacc << vk) | (vv & ((1ull << vk) - 1)); 
            van += vk;
            rle_flush_value_mantissa(vman, vman_capacity, vbp, vacc, van);
        }
        if (ran > 0) rman[rbp++] = (racc << (8 - ran));
        if (van > 0) vman[vbp++] = (vacc << (8 - van));

        uchar *start = bytes;
        bytes += 6 * sizeof(uint64_t);
        size_t s_rcl = fse_enc(rcl, nr, bytes); bytes += s_rcl;
        size_t s_vcl = fse_enc(vcl, nv, bytes); bytes += s_vcl;
        memcpy(bytes, rman, rbp);    bytes += rbp;
        memcpy(bytes, vman, vbp);    bytes += vbp;
        uchar *header = start;
        write(static_cast<uint64_t>(nr), header); write(static_cast<uint64_t>(nv), header);
        write(static_cast<uint64_t>(s_rcl), header); write(static_cast<uint64_t>(s_vcl), header);
        write(static_cast<uint64_t>(rbp), header);  write(static_cast<uint64_t>(vbp), header);
        return static_cast<size_t>(bytes - start);
    }

    T *decode(const uchar *&bytes, size_t N) override {
        T *out;
        // static const bool fse_ok = o1_fse_selftest();   // link self-test (tableLog vs linked FSE); uncomment to re-arm
        // (void)fse_ok;
        if (method_ == 2) {
            constexpr size_t pad = 64 / sizeof(T);
            if (N > (std::numeric_limits<size_t>::max)() / sizeof(T) - pad)
                throw std::length_error("byte split decoded size overflow");
            std::unique_ptr<T[]> decoded(new T[N + pad]);
            EncodeWorkspace &ws = encode_workspace();
            decode_split_fse(bytes, decode_remaining_, N, decoded.get(), ws.byte_low.get(FBLK),
                             ws.byte_high.get(FBLK), ws.byte_fse_workspace());
            std::fill_n(decoded.get() + N, pad, T{});
            out = decoded.release();
        } else if (method_ == 1) out = decode_h1fse(bytes, N);
        // else if (method_ == 2) out = decode_rle_adt(bytes, N);
        // else if (method_ == 3) out = decode_adt(bytes, N);
        else {
        const T mode = mode_;
        const int32_t modev = mode_;

        uint64_t nr, nv, s_rcl, s_vcl, rbp, vbp;
        read(nr, bytes); 
        read(nv, bytes); 
        read(s_rcl, bytes); 
        read(s_vcl, bytes); 
        read(rbp, bytes); 
        read(vbp, bytes);
        const uchar *o_rcl = bytes; bytes += s_rcl;
        const uchar *o_vcl = bytes; bytes += s_vcl;
        const uchar *rman = bytes;  bytes += rbp;
        const uchar *vman = bytes;  bytes += vbp;

        uint8_t *drcl = static_cast<uint8_t*>(malloc(nr ? nr : 1)), *dvcl = static_cast<uint8_t*>(malloc(nv ? nv : 1));
        
        fse_dec(o_rcl, nr, drcl); 
        if (nv) 
            fse_dec(o_vcl, nv, dvcl);

        constexpr size_t pad = 64 / sizeof(T);
        out = new T[N + pad];
#if defined(__linux__)
        {   // hint THP for the decoded quant array (large hot buffer; big decompress win)
            uintptr_t _mb = reinterpret_cast<uintptr_t>(out) & ~static_cast<uintptr_t>(4095);
            madvise(reinterpret_cast<void *>(_mb), (N + pad) * sizeof(T) + (reinterpret_cast<uintptr_t>(out) - _mb), MADV_HUGEPAGE);
        }
#endif
        size_t prod = 0, rrp = 0, vrp = 0, ri = 0, vi = 0;
        uint64_t rda = 0, vda = 0; 
        int rdn = 0, vdn = 0;
        while (prod < N) {
            int k = drcl[ri++];
            while (rdn < k) { 
                rda = (rda << 8) | rman[rrp++]; 
                rdn += 8; 
            }
            rdn -= k; 
            uint64_t v = (1ull << k) | ((rda >> rdn) & ((1ull << k) - 1));
            size_t L = (v - 1);
            if (L > N - prod) L = N - prod;
            fill_mode(out + prod, L, mode); 
            prod += L;
            if (prod >= N) 
                break;
            int vk = dvcl[vi++];
            while (vdn < vk) { 
                vda = (vda << 8) | vman[vrp++]; 
                vdn += 8; 
            }
            vdn -= vk; 
            uint64_t vv = (1ull << vk) | ((vda >> vdn) & ((1ull << vk) - 1));
            uint32_t zz = vv - 1; 
            int32_t fct = static_cast<int32_t>((zz >> 1) ^ (~(zz & 1) + 1));
            out[prod++] = static_cast<T>(modev + fct);
        }
        free(drcl);
        free(dvcl);
        }
        if (use_delta_ && N > 0) {   // un-delta = prefix sum in SIMD blocks plus a running carry
            size_t i = 1;
#ifdef __AVX2__
            __m128i carry = _mm_set1_epi16(static_cast<int16_t>(out[0]));
            for (; i + 8 <= N; i += 8) {
                __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(out + i));
                v = _mm_add_epi16(v, _mm_slli_si128(v, 2));   // in-block prefix sum: +1,+2,+4 elem
                v = _mm_add_epi16(v, _mm_slli_si128(v, 4));
                v = _mm_add_epi16(v, _mm_slli_si128(v, 8));
                v = _mm_add_epi16(v, carry);                  // + running total from prior blocks
                _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), v);
                carry = _mm_shuffle_epi8(v, _mm_set1_epi16(0x0F0E));   // broadcast lane 7 = new running total
            }
#endif
            for (; i < N; i++) 
                out[i] = static_cast<T>(out[i - 1] + out[i]);   // scalar tail (+head if no SIMD)
        }
        return out;
    }

    void save(uchar *&c) override {
        write(static_cast<uint8_t>(method_ | (use_delta_ ? 0x80 : 0)), c);
        write(lo_, c); write(mode_, c);
    }
    void load(const uchar *&c, size_t &remaining_length) override {
        if (remaining_length < sizeof(uint8_t) + 2 * sizeof(uint16_t))
            throw std::runtime_error("truncated entropy header");
        read(method_, c, remaining_length); use_delta_ = (method_ & 0x80) != 0; method_ &= 0x7F;
        read(lo_, c, remaining_length); read(mode_, c, remaining_length);
        decode_remaining_ = remaining_length;
    }

    void preprocess_encode(const std::vector<T> &bins, int stateNum) override {
        preprocess_encode(bins.data(), bins.size(), stateNum);
    }
    size_t encode(const std::vector<T> &bins, uchar *&bytes) override {
        return encode(bins.data(), bins.size(), bytes);
    }
    void preprocess_decode() override {}
    void postprocess_encode() override {
        // Release per-call scratch before final Zstd; keep the serialized model flags.
        workspace_.reset();
    }
    void postprocess_decode() override {}
    size_t size_est() override { return 64; }
    // Fixed per-call overhead the encoder can emit regardless of input size: FSE normalized-count
    // tables for the RLE+FSE path (method 0) plus headers. (The order-1 tANS path's much larger
    // ~15.9 KB table is gated to large inputs in preprocess_encode, where sizeof(T)*num_bin already
    // dwarfs it.) The compress buffer is sized from this, so it must cover method 0 on tiny/
    // incompressible inputs or the encoder overflows it before the lossless-fallback check.
    size_t size_est_without_init() override { return 4096; }
};

}  // namespace Falcons
#endif  // Falcons_RLE_FSE_ENCODER_HPP
