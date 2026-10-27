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
 * RleFseEncoder — run-length + FSE entropy coder replacing Huffman in Falcons-BR2.
 *
 * Falcons interpolation quant streams are extremely peaked: the zero-error code
 * ("mode") covers ~99% of symbols and forms long spatial runs. This coder:
 *
 *   1. SIMD-scans mode runs (AVX2 or SVE2) — work scales with the number
 *      of runs (~1% of N at loose bounds), not with N.
 *   2. Reparametrizes the stream as alternating [run-length L][non-mode value v].
 *   3. Codes L and zigzag(v-mode) each as Elias-gamma: a magnitude-class byte
 *      (FSE-compressed, near-entropy) plus k raw mantissa bits.
 *
 * Unlike a plain order-0 coder (FSE/ANS), the explicit run capture models the
 * run structure (order-1), so it matches/beats Huffman+zstd on ratio; unlike
 * the adaptive binary range coder, it uses fast static FSE table coding, so it
 * is faster than both. The downstream zstd stage is kept (near-neutral here).
 *
 * Hybrid: when p(mode) < PMODE_MIN the runs degenerate (short) and RLE no longer
 * pays off — fall back to ADT-FSE (Int2code transcode + per-block FSE + bucket-diff
 * bits), which stays fast and near-Huffman on ratio at tight bounds.
 *
 * Only external dependency: FSE_compress/FSE_decompress (exported by the libzstd
 * the pipeline already links). No zstd headers needed.
 */

#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

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
#include "zfse/fse.h"   // low-level FSE: buildCTable/DTable + inline encodeSymbol/decodeSymbol + BIT stream (zstd 1.4, matches libzstd 1.4.8)
extern "C" {
size_t FSE_buildCTable(FSE_CTable *ct, const short *normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
size_t FSE_buildDTable(FSE_DTable *dt, const short *normalizedCounter, unsigned maxSymbolValue, unsigned tableLog);
}

namespace Falcons {

template <class T>
class RleFseEncoder : public concepts::EncoderInterface<T> {
    static_assert(sizeof(T) == 2, "RleFseEncoder expects 16-bit quant codes");

    static constexpr double PMODE_MIN = 0;   // p(mode)>=this -> RLE+FSE (loose); below -> ADT-FSE (tight)
    // FSE block size: per-block FSE table + raw/RLE fallback (ADT2-FSE). 64K.
    static constexpr int FBLK = 1 << 16;

    uint16_t lo_ = 0, mode_ = 0;
    uint8_t method_ = 0;                         // 0=RLE+FSE 1=tANS 2=RLE+ADT2 3=ADT2
    bool use_delta_ = false;                     // forward-diff delta (decided by dzeros)
    bool scan_simd_ = true;                      // SIMD in scan_run (decided by p0>=0.85)

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
    static size_t fse_enc(const uint8_t *src, size_t n, uint8_t *out) {
        size_t p = 0; 
        uint8_t *tmp = static_cast<uint8_t*>(malloc(FBLK + 1024));
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
        free(tmp);
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

    // ADT transcode (from Falcons_ADT): factor -> 67-code class (FSE) + bucket-diff bits. Finer
    // than gamma near 0 (each |factor|<=15 is its own FSE symbol), so better at tight bounds.
    // static constexpr uint8_t Ft_Code[64] = {
    //     0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,16,17,17,18,18,19,19,
    //     20,20,20,20,21,21,21,21,22,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,
    //     24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24};
    // static constexpr int code2int[67][2] = {
    //     {0,0},{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0},{9,0},{10,0},{11,0},
    //     {12,0},{13,0},{14,0},{15,0},{16,1},{18,1},{20,1},{22,1},{24,2},{28,2},{32,3},{40,3},
    //     {48,4},{64,6},{128,7},{256,8},{512,9},{1024,10},{2048,11},{4096,12},{8192,13},{16384,14},
    //     {-16384,14},{-8192,13},{-4096,12},{-2048,11},{-1024,10},{-512,9},{-256,8},{-128,7},{-64,6},{-48,4},
    //     {-40,3},{-32,3},{-28,2},{-24,2},{-22,1},{-20,1},{-18,1},{-16,1},{-15,0},{-14,0},{-13,0},{-12,0},
    //     {-11,0},{-10,0},{-9,0},{-8,0},{-7,0},{-6,0},{-5,0},{-4,0},{-3,0},{-2,0},{-1,0}};
    // static inline int Int2code(int f) {
    //     if (f >= 0) return (f > 63) ? 50 - __builtin_clz(f) : Ft_Code[f];
    //     f = -f; return (f > 63) ? 17 + __builtin_clz(f) : 67 - Ft_Code[f];
    // }
    // struct AdtTab { uint8_t t2c[65536]; uint16_t diff[65536]; uint8_t nb[65536]; };
    // static const AdtTab &adt_tab() {   // precompute over int16 codes once (thread-safe). index = (uint16)code.
    //     static const AdtTab *t = []() {
    //         AdtTab *p = new AdtTab();
    //         for (int u = 0; u < 65536; u++) {
    //             int factor = (int16_t)u;            // int16 quant code IS the factor (mode = 0)
    //             if (factor == -32768) continue;     // escape, handled separately
    //             int c = Int2code(factor);
    //             p->t2c[u] = (uint8_t)c; int base = code2int[c][0]; int d = factor - base; if (d < 0) d = -d;
    //             p->diff[u] = (uint16_t)d; p->nb[u] = (uint8_t)code2int[c][1];
    //         }
    //         return p;
    //     }();
    //     return *t;
    // }

    // // ADT-FSE path (method_==1): int16 quant code = factor -> Int2code class (FSE) + bucket-diff mantissa.
    // size_t encode_adt(const T *bins, size_t N, uchar *&bytes) {
    //     const AdtTab &tab = adt_tab();
    //     uint8_t *tpc = (uint8_t *)malloc(N), *man = (uint8_t *)malloc(2 * N + 16);
    //     size_t mbp = 0; uint64_t acc = 0; int an = 0;
    //     for (size_t i = 0; i < N; i++) {
    //         int code = (int)bins[i];
    //         if (code == -32768) { tpc[i] = 67; }            // escape (unpredictable)
    //         else {
    //             uint16_t u = (uint16_t)code;                // reinterpret signed code -> table index
    //             tpc[i] = tab.t2c[u]; int nb = tab.nb[u];
    //             if (nb) { acc = (acc << nb) | tab.diff[u]; an += nb; while (an >= 8) { an -= 8; man[mbp++] = (uint8_t)(acc >> an); } }
    //         }
    //     }
    //     if (an > 0) man[mbp++] = (uint8_t)(acc << (8 - an));
    //     uint8_t *o_tpc = (uint8_t *)malloc(N + N / FBLK * 8 + 64);
    //     size_t s_tpc = fse_enc(tpc, N, o_tpc);
    //     uchar *start = bytes;
    //     write((uint64_t)N, bytes); write((uint64_t)s_tpc, bytes); write((uint64_t)mbp, bytes);
    //     memcpy(bytes, o_tpc, s_tpc); bytes += s_tpc;
    //     memcpy(bytes, man, mbp);    bytes += mbp;
    //     free(tpc); free(man); free(o_tpc);
    //     return (size_t)(bytes - start);
    // }
    // T *decode_adt(const uchar *&bytes, size_t N) {
    //     uint64_t ntp, s_tpc, mbp;
    //     read(ntp, bytes); read(s_tpc, bytes); read(mbp, bytes);
    //     const uchar *o_tpc = bytes; bytes += s_tpc;
    //     const uchar *man = bytes;   bytes += mbp;
    //     uint8_t *dtpc = (uint8_t *)malloc(ntp ? ntp : 1);
    //     fse_dec(o_tpc, ntp, dtpc);
    //     constexpr size_t pad = 64 / sizeof(T);
    //     T *out = new T[N + pad];
    //     size_t rp = 0; uint64_t da = 0; int dn = 0;
    //     for (size_t i = 0; i < N; i++) {
    //         int c = dtpc[i];
    //         if (c == 67) { out[i] = (T)(-32768); continue; }   // escape
    //         int base = code2int[c][0], nb = code2int[c][1], diff = 0;
    //         if (nb) { while (dn < nb) { da = (da << 8) | man[rp++]; dn += 8; } dn -= nb; diff = (int)((da >> dn) & ((1u << nb) - 1)); }
    //         int factor = (c < 34) ? base + diff : base - diff;
    //         out[i] = (T)factor;                                // int16: code = factor (mode = 0)
    //     }
    //     free(dtpc);
    //     return out;
    // }

    // // RLE+ADT (method_==2): mode runs coded as gamma run-length (same as RLE+FSE), but each non-mode
    // // value uses the ADT transcode (Int2code class FSE'd + bucket-diff mantissa) instead of gamma.
    // // Per-block FSE (FBLK-sized blocks, RLE+ADT2) on the value-class stream.
    // size_t encode_rle_adt(const T *bins, size_t N, uchar *&bytes) {
    //     const AdtTab &tab = adt_tab();
    //     const T mode = (T)mode_;
    //     uint8_t *rcl = (uint8_t *)malloc(N + 1), *vcl = (uint8_t *)malloc(N + 1);
    //     uint8_t *rman = (uint8_t *)malloc(N + 16), *vman = (uint8_t *)malloc(2 * N + 16);
    //     size_t nr = 0, nv = 0, rbp = 0, vbp = 0; uint64_t racc = 0, vacc = 0; int ran = 0, van = 0;
    //     size_t i = 0;
    //     while (i < N) {
    //         size_t L = scan_run(bins, i, N, mode, scan_avx_); i += L;
    //         uint64_t v = (uint64_t)L + 1; int k = 63 - __builtin_clzll(v); rcl[nr++] = (uint8_t)k;
    //         racc = (racc << k) | (v & ((k < 64) ? ((1ull << k) - 1) : ~0ull)); ran += k;
    //         while (ran >= 8) { ran -= 8; rman[rbp++] = (uint8_t)(racc >> ran); }
    //         if (i >= N) break;
    //         int code = (int)bins[i++];
    //         if (code == -32768) { vcl[nv++] = 67; }                       // escape (unpredictable)
    //         else {
    //             uint16_t u = (uint16_t)code;
    //             vcl[nv++] = tab.t2c[u]; int nb = tab.nb[u];
    //             if (nb) { vacc = (vacc << nb) | tab.diff[u]; van += nb; while (van >= 8) { van -= 8; vman[vbp++] = (uint8_t)(vacc >> van); } }
    //         }
    //     }
    //     if (ran > 0) rman[rbp++] = (uint8_t)(racc << (8 - ran));
    //     if (van > 0) vman[vbp++] = (uint8_t)(vacc << (8 - van));

    //     uint8_t *o_rcl = (uint8_t *)malloc(nr + nr / FBLK * 8 + 64);
    //     uint8_t *o_vcl = (uint8_t *)malloc(nv + nv / FBLK * 8 + 64);
    //     size_t s_rcl = fse_enc(rcl, nr, o_rcl), s_vcl = fse_enc(vcl, nv, o_vcl);

    //     uchar *start = bytes;
    //     write((uint64_t)nr, bytes); write((uint64_t)nv, bytes);
    //     write((uint64_t)s_rcl, bytes); write((uint64_t)s_vcl, bytes);
    //     write((uint64_t)rbp, bytes);  write((uint64_t)vbp, bytes);
    //     memcpy(bytes, o_rcl, s_rcl); bytes += s_rcl;
    //     memcpy(bytes, o_vcl, s_vcl); bytes += s_vcl;
    //     memcpy(bytes, rman, rbp);    bytes += rbp;
    //     memcpy(bytes, vman, vbp);    bytes += vbp;
    //     free(rcl); free(vcl); free(rman); free(vman); free(o_rcl); free(o_vcl);
    //     return (size_t)(bytes - start);
    // }

    // T *decode_rle_adt(const uchar *&bytes, size_t N) {
    //     const T mode = (T)mode_;
    //     uint64_t nr, nv, s_rcl, s_vcl, rbp, vbp;
    //     read(nr, bytes); read(nv, bytes); read(s_rcl, bytes); read(s_vcl, bytes); read(rbp, bytes); read(vbp, bytes);
    //     const uchar *o_rcl = bytes; bytes += s_rcl;
    //     const uchar *o_vcl = bytes; bytes += s_vcl;
    //     const uchar *rman = bytes;  bytes += rbp;
    //     const uchar *vman = bytes;  bytes += vbp;
    //     uint8_t *drcl = (uint8_t *)malloc(nr ? nr : 1), *dvcl = (uint8_t *)malloc(nv ? nv : 1);
    //     fse_dec(o_rcl, nr, drcl); if (nv) fse_dec(o_vcl, nv, dvcl);
    //     constexpr size_t pad = 64 / sizeof(T);
    //     T *out = new T[N + pad];
    //     size_t prod = 0, rrp = 0, vrp = 0, ri = 0, vi = 0; uint64_t rda = 0, vda = 0; int rdn = 0, vdn = 0;
    //     while (prod < N) {
    //         int k = drcl[ri++];
    //         while (rdn < k) { rda = (rda << 8) | rman[rrp++]; rdn += 8; }
    //         rdn -= k; uint64_t v = (1ull << k) | ((rda >> rdn) & ((k < 64) ? ((1ull << k) - 1) : ~0ull));
    //         size_t L = (size_t)(v - 1);
    //         if (L > N - prod) L = N - prod;
    //         for (size_t j = 0; j < L; j++) out[prod++] = mode;
    //         if (prod >= N) break;
    //         int c = dvcl[vi++];
    //         if (c == 67) { out[prod++] = (T)(-32768); continue; }         // escape
    //         int base = code2int[c][0], nb = code2int[c][1], diff = 0;
    //         if (nb) { while (vdn < nb) { vda = (vda << 8) | vman[vrp++]; vdn += 8; } vdn -= nb; diff = (int)((vda >> vdn) & ((1u << nb) - 1)); }
    //         int factor = (c < 34) ? base + diff : base - diff;
    //         out[prod++] = (T)factor;
    //     }
    //     free(drcl); free(dvcl);
    //     return out;
    // }

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

    // static bool o1_fse_selftest() {
    //     int16_t norm[2];
    //     norm[0] = 1;
    //     norm[1] = static_cast<int16_t>(O1T_M - 1);
    //     FSE_CTable *ct = static_cast<FSE_CTable *>(malloc(FSE_CTABLE_SIZE_U32(O1T_PB, 1) * sizeof(FSE_CTable)));
    //     size_t r = FSE_buildCTable(ct, norm, 1, O1T_PB);
    //     free(ct);
    //     if (FSE_isError(r)) {
    //         std::fprintf(stderr,
    //             "Falcons o1tans: FSE_buildCTable self-test failed: tableLog %d exceeds the linked FSE "
    //             "build. This binary must compile the vendored zfse sources (see "
    //             "tools/falcons/CMakeLists.txt), not resolve FSE_* from libzstd.\n", O1T_PB);
    //         std::abort();
    //     }
    //     return true;
    // }

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
    // ---- K-stream context-switching tANS (method_==4) ----
    // K independent FSE bitstreams (chunks), each switching (stateTable,symbolTT)/DTable per symbol
    // by the 15-way context. Interleaved -> K independent chains hide the FSE state latency (ILP).
    static size_t encode_o1_ovf(const int16_t *ovf, size_t n, uchar *&bytes) {
        uchar *sb = bytes;
        if (n == 0) {
            write(static_cast<uint8_t>(0), bytes);
            return bytes - sb;
        }

        uint8_t *vcl = static_cast<uint8_t *>(malloc(n));
        uint8_t *vman = static_cast<uint8_t *>(malloc(2 * n + 16));
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
            while (van >= 8) {
                van -= 8;
                vman[vbp++] = static_cast<uint8_t>(vacc >> van);
            }
        }
        if (van > 0)
            vman[vbp++] = static_cast<uint8_t>(vacc << (8 - van));

        uint8_t *o_vcl = static_cast<uint8_t *>(malloc(n + n / FBLK * 8 + 64));
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
        free(vcl);
        free(vman);
        free(o_vcl);
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
    size_t encode_o1tans(const T *bins, size_t N, uchar *&bytes) {
        uchar *sb = bytes;
        size_t S = N / O1T_SEG;
        if (S < 1) S = 1;
        if (S > 16) S = 16;
        write(static_cast<uint64_t>(S), bytes);
        size_t seg = (N + S - 1) / S;
        for (size_t g = 0; g < S; g++) {
            size_t lo = g * seg, hi = lo + seg;
            if (hi > N) hi = N;
            encode_o1tans_block(bins + lo, hi - lo, bytes);
        }
        return static_cast<size_t>(bytes - sb);
    }

    size_t encode_o1tans_block(const T *bins, size_t N, uchar *&bytes) {
        const O1Lut &L = o1_lut();
        uchar *sb = bytes;
        // if (N == 0) { 
        //     write((uint64_t)0, bytes); 
        //     write((uint64_t)O1T_K, bytes); 
        //     return (size_t)(bytes - sb); 
        // }
        size_t cs = (N + O1T_K - 1) / O1T_K;
        int16_t *ovf = static_cast<int16_t *>(malloc(N * 2));
        size_t novf = 0;
        size_t novf_k[16] = {0};
        uint32_t *cnt = static_cast<uint32_t *>(calloc(O1_NCTX * O1T_NSYM, 4));
        //std::getenv("FALCONS_TANS_FULLHIST") || 
        if (N < (1u << 20)) {       // exact full per-context histogram (opt-out via env / small N); default = fast sampled path below
            for (int k = 0; k < O1T_K; k++) { 
                size_t lo = cs * k, hi = lo + cs; 
                if (hi > N) 
                    hi = N;
                uint16_t prev = 0;
                for (size_t i = lo; i < hi; i++) { 
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
        } else {                                                        // fast: full escape scan + windowed-sample histogram
            for (int k = 0; k < O1T_K; k++) { 
                size_t lo = cs * k, hi = lo + cs; 
                if (hi > N) 
                    hi = N; 
                size_t b0 = novf;  // escapes: branchless
                for (size_t i = lo; i < hi; i++) { 
                    ovf[novf] = bins[i];
                    novf += (L.sym256[static_cast<uint16_t>(bins[i])] == O1T_NSYM - 1);
                }
                novf_k[k] = novf - b0; 
            }
            size_t nwin = 2048, wlen = 512;                             // ~1M windowed pairs per block; prev exact within each window
            size_t step = N / nwin; 
            if (step < wlen) 
                step = wlen;
            for (size_t w = 0; w < N; w += step) { 
                int prev = (w > 0) ? bins[w - 1] : 0; 
                size_t we = w + wlen; 
                if (we > N) 
                    we = N;
                for (size_t i = w; i < we; i++) { 
                    int code = bins[i];
                    cnt[static_cast<size_t>(O1T_NSYM) * L.ctx[static_cast<uint16_t>(prev)] + L.sym256[static_cast<uint16_t>(code)]]++;
                    prev = code; 
                } 
            }
            size_t total = O1_NCTX * O1T_NSYM;
            for (size_t t = 0; t < total; t++) 
                cnt[t]++;   // Laplace +1: no zero-freq -> any real symbol stays encodable
        }
        
        uint16_t *nf = static_cast<uint16_t *>(malloc(O1_NCTX * O1T_NSYM * 2));

        const void *stTab[O1_NCTX];
        const void *syTT[O1_NCTX];
        size_t ctsz = FSE_CTABLE_SIZE_U32(O1T_PB, O1T_NSYM - 1);
        FSE_CTable *ct[O1_NCTX];
        FSE_CTable *ct_storage = static_cast<FSE_CTable *>(malloc(O1_NCTX * ctsz * sizeof(FSE_CTable)));

        int16_t normc[O1T_NSYM];
        for (int c = 0; c < O1_NCTX; c++) {
            o1_normalize(cnt + c * O1T_NSYM, nf + c * O1T_NSYM, O1T_NSYM, O1T_M);
            for (int s = 0; s < O1T_NSYM; s++) 
                normc[s] = nf[c * O1T_NSYM + s];
            ct[c] = ct_storage + static_cast<size_t>(c) * ctsz;
            FSE_buildCTable(ct[c], normc, O1T_NSYM - 1, O1T_PB);
            FSE_CState_t tmp; 
            FSE_initCState(&tmp, ct[c]); 
            stTab[c] = tmp.stateTable; 
            syTT[c] = tmp.symbolTT;
        }
        uint8_t *buf[16]; 
        BIT_CStream_t bc[16]; 
        FSE_CState_t cst[16]; 
        size_t initIdx[16]; 
        size_t capk = cs * 2 + 4096;
        for (int k = 0; k < O1T_K; k++) {
            buf[k] = static_cast<uint8_t *>(malloc(capk)); 
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
        for (size_t jj = cs; jj-- > 0; ) {                                 // interleaved reverse; O1T_K independent chains -> ILP
            for (int k = 0; k < O1T_K; k++) { 
                size_t idx = k * cs + jj; 
                if (idx >= N || idx == initIdx[k]) 
                    continue;
                size_t lo = k * cs; 
                int prevc = (idx > lo) ? bins[idx - 1] : 0;
                int c = L.ctx[static_cast<uint16_t>(prevc)]; 
                int s = L.sym256[static_cast<uint16_t>(bins[idx])];
                cst[k].stateTable = stTab[c]; 
                cst[k].symbolTT = syTT[c];
                FSE_encodeSymbol(&bc[k], &cst[k], static_cast<int16_t>(s)); 
            }
            if ((jj & 3) == 0) 
                for (int k = 0; k < O1T_K; k++) 
                    BIT_flushBits(&bc[k]); 
        }  // flush every 4 (4*11<57)
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
        for (int k = 0; k < O1T_K; k++) 
            free(buf[k]);
        free(ct_storage); free(ovf); free(cnt); free(nf);
        return (bytes - sb);
    }

    T *decode_o1tans(const uchar *&bytes, size_t N) {
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
            decode_o1tans_block(bytes, hi - lo, out + lo);
        }
        return out;
    }

    void decode_o1tans_block(const uchar *&bytes, size_t N, T *out) {
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
    // void preprocess_encode(const T *bins, size_t num_bin, int stateNum, size_t *frequencyList) {
    //     (void)frequencyList;               // no histogram needed (mode = radius, known a priori)
    //     derive(bins, num_bin, stateNum);
    // }
    void preprocess_encode(const T *bins, size_t num_bin, int stateNum) {
        // static const bool fse_ok = o1_fse_selftest();   // link self-test (tableLog vs linked FSE); uncomment to re-arm
        // (void)fse_ok;
        // (void)stateNum;
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
        for (size_t w = 0; w < NW; w++) {
            size_t off = w * step; 
            if (off + WL > num_bin) 
                break;
            T prev = bins[off];
            for (size_t t = 0; t < WL; t++) {
                T x = bins[off + t];
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
        scan_simd_ = (p0 >= 0.85);
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
        if (method_ == 1) return encode_o1tans(bins, N, bytes);   // 1 = order1-tANS
        // if (method_ == 2) return encode_rle_adt(bins, N, bytes);  // 2 = RLE+ADT2+zstd
        // if (method_ == 3) return encode_adt(bins, N, bytes);      // 3 = ADT2+zstd (noRLE)
        // method_ == 0 -> RLE+FSE+zstd (fallthrough)
        const T mode = static_cast<T>(mode_);
        const int32_t modev = static_cast<int32_t>(mode_);
        // scratch — raw malloc (NOT zero-initialized): we only touch the bytes we
        // actually write, so no 5N memset cost on the hot path.
        uint8_t *rcl = static_cast<uint8_t*>(malloc(N + 1)), *vcl = static_cast<uint8_t*>(malloc(N + 1));
        uint8_t *rman = static_cast<uint8_t*>(malloc(N + 16)), *vman = static_cast<uint8_t*>(malloc(2 * N + 16));
        size_t nr = 0, nv = 0, rbp = 0, vbp = 0; 
        uint64_t racc = 0, vacc = 0; 
        int ran = 0, van = 0;
        size_t i = 0;
        while (i < N) {
            size_t L = scan_run(bins, i, N, mode, scan_simd_);
            i += L;
            uint64_t v = L + 1; 
            int k = 63 - __builtin_clzll(v); 
            rcl[nr++] = k;
            racc = (racc << k) | (v & ((1ull << k) - 1)); 
            ran += k;
            while (ran >= 8) { 
                ran -= 8; 
                rman[rbp++] = (racc >> ran); 
            }

            if (i >= N) break;

            int32_t fct = bins[i++] - modev; 
            uint32_t zz = (static_cast<uint32_t>(fct) << 1) ^ static_cast<uint32_t>(fct >> 31);
            uint64_t vv = static_cast<uint32_t>(zz) + 1; 
            int vk = 63 - __builtin_clzll(vv); 
            vcl[nv++] = vk;
            vacc = (vacc << vk) | (vv & ((1ull << vk) - 1)); 
            van += vk;
            while (van >= 8) {
                van -= 8; 
                vman[vbp++] = vacc >> van; 
            }
        }
        if (ran > 0) rman[rbp++] = (racc << (8 - ran));
        if (van > 0) vman[vbp++] = (vacc << (8 - van));

        uint8_t *o_rcl = static_cast<uint8_t*>(malloc(nr + nr / FBLK * 8 + 64));
        uint8_t *o_vcl = static_cast<uint8_t*>(malloc(nv + nv / FBLK * 8 + 64));

        size_t s_rcl = fse_enc(rcl, nr, o_rcl), s_vcl = fse_enc(vcl, nv, o_vcl);
        uchar *start = bytes;
        write(static_cast<uint64_t>(nr), bytes); write(static_cast<uint64_t>(nv), bytes);
        write(static_cast<uint64_t>(s_rcl), bytes); write(static_cast<uint64_t>(s_vcl), bytes);
        write(static_cast<uint64_t>(rbp), bytes);  write(static_cast<uint64_t>(vbp), bytes);
        memcpy(bytes, o_rcl, s_rcl); bytes += s_rcl;
        memcpy(bytes, o_vcl, s_vcl); bytes += s_vcl;
        memcpy(bytes, rman, rbp);    bytes += rbp;
        memcpy(bytes, vman, vbp);    bytes += vbp;
        free(rcl); free(vcl); free(rman); free(vman); free(o_rcl); free(o_vcl);
        return static_cast<size_t>(bytes - start);
    }

    T *decode(const uchar *&bytes, size_t N) override {
        T *out;
        // static const bool fse_ok = o1_fse_selftest();   // link self-test (tableLog vs linked FSE); uncomment to re-arm
        // (void)fse_ok;
        if (method_ == 1) out = decode_o1tans(bytes, N);
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
        read(method_, c, remaining_length); use_delta_ = (method_ & 0x80) != 0; method_ &= 0x7F;
        read(lo_, c, remaining_length); read(mode_, c, remaining_length);
    }

    void preprocess_encode(const std::vector<T> &bins, int stateNum) override {
        preprocess_encode(bins.data(), bins.size(), stateNum);
    }
    size_t encode(const std::vector<T> &bins, uchar *&bytes) override {
        return encode(bins.data(), bins.size(), bytes);
    }
    void preprocess_decode() override {}
    void postprocess_encode() override {}
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
