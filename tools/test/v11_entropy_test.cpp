// Standalone V1.1 entropy correctness tests; link the vendored Falcons FSE archive.
// Define FALCONS_V11_TEST_ZSTD and link Zstd to also wrap each entropy payload.
#ifndef FALCONS_V11_ENTROPY_TEST
#define FALCONS_V11_ENTROPY_TEST
#endif
#include "Falcons/encoder/RleFseEncoder.hpp"
#ifdef FALCONS_V11_TEST_ZSTD
#include "Falcons/lossless/Lossless_zstd.hpp"
#endif

#include <array>
#include <atomic>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace {
void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

uint32_t next_random(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

std::atomic<size_t> round_trips{0};
}

namespace Falcons {
struct RleFseEncoderV11TestAccess {
    template <class T>
    static void force(RleFseEncoder<T> &e, int method, bool delta, bool simd) {
        e.method_ = static_cast<uint8_t>(method);
        e.use_delta_ = delta;
        e.scan_simd_ = simd;
    }

    template <class T>
    static const void *workspace(const RleFseEncoder<T> &e) {
        return e.workspace_.get();
    }

    template <class T>
    static void check_scratch() {
        using E = RleFseEncoder<T>;
        E e;
        auto &ws = e.encode_workspace();
        for (size_t n : {size_t{0}, size_t{1}, size_t{15}, size_t{16}, size_t{17},
                         size_t{63}, size_t{64}, size_t{65}, size_t{4096}, size_t{80001}}) {
            const auto b = ws.rle_buffers(n);
            for (const uint8_t *p : {b.rcl, b.vcl, b.rman, b.vman})
                check(reinterpret_cast<uintptr_t>(p) % 64 == 0, "RLE scratch alignment");
            check(b.vcl - b.rcl >= static_cast<ptrdiff_t>(n + 1), "RLE class overlap");
            check(b.rman - b.vcl >= static_cast<ptrdiff_t>(n + 1), "RLE value overlap");
            check(b.vman - b.rman >= static_cast<ptrdiff_t>(n + 16), "RLE mantissa overlap");
            check(b.vman_capacity == (17 * n + 7) / 8 + 16, "RLE 17-bit mantissa capacity");
            check(b.vman + b.vman_capacity <= ws.rle_slab.data.get() + ws.rle_slab.capacity,
                  "RLE slab bounds");
        }
        const auto first = ws.rle_buffers(4096);
        check(first.rcl == ws.rle_buffers(17).rcl, "RLE slab not reused");
        auto *tmp = ws.fse_tmp.get(E::FBLK + 1024);
        check(tmp == ws.fse_tmp.get(E::FBLK + 1024), "FSE scratch not reused");
        for (size_t n : {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max() / 2}) {
            bool rejected = false;
            try { (void)E::rle_layout(n); }
            catch (const std::bad_array_new_length &) { rejected = true; }
            check(rejected, "RLE layout arithmetic overflow");
            rejected = false;
            try { (void)E::rle_vman_capacity(n); }
            catch (const std::bad_array_new_length &) { rejected = true; }
            check(rejected, "RLE mantissa capacity arithmetic overflow");
        }
        force(e, 1, true, false);
        E copied(e), assigned;
        assigned.encode_workspace().rle_buffers(17);
        assigned = e;
        check(!workspace(copied) && !workspace(assigned), "Encoder copy shared scratch");
        check(copied.method_ == 1 && copied.use_delta_ && !copied.scan_simd_ &&
              assigned.method_ == 1 && assigned.use_delta_ && !assigned.scan_simd_,
              "Encoder copy lost model flags");
        const void *saved = workspace(e);
        E moved(std::move(e));
        check(workspace(moved) == saved && !workspace(e), "Encoder move lost scratch");
        e = std::move(moved);
        check(workspace(e) == saved && !workspace(moved), "Encoder move assignment");
        e = e;
        check(workspace(e) == saved, "Encoder self assignment");
        e.postprocess_encode();
        check(!workspace(e), "Encoder scratch not released");
    }

    static void check_mantissa() {
        using E = RleFseEncoder<int16_t>;
        for (int bits = 0; bits <= 24; ++bits) {
            for (uint64_t acc : {uint64_t{0}, uint64_t{0x123456789abcdef0ULL}, ~uint64_t{0}}) {
                for (bool bulk : {false, true}) {
                    std::array<uint8_t, 16> out;
                    out.fill(0xa5);
                    size_t pos = 1;
                    int remaining = bits;
                    const size_t capacity = bulk ? 9 : 1 + static_cast<size_t>(bits / 8);
                    E::rle_flush_value_mantissa(out.data(), capacity, pos, acc, remaining);
                    check(pos == 1 + static_cast<size_t>(bits / 8), "Mantissa byte count");
                    check(remaining == (bits & 7), "Mantissa carry count");
                    for (int j = 0; j < bits / 8; ++j)
                        check(out[1 + j] == static_cast<uint8_t>(acc >> (bits - 8 * (j + 1))),
                              "Mantissa packing");
                    check(out[0] == 0xa5, "Mantissa prefix guard");
                    for (size_t j = capacity; j < out.size(); ++j)
                        check(out[j] == 0xa5, "Mantissa capacity guard");
                }
            }
        }
    }

#ifdef __AVX2__
    template <class T>
    static void check_avx2() {
        using E = RleFseEncoder<T>;
        const T mode = 0;
        const __m256i vm = _mm256_setzero_si256();
        for (unsigned pattern = 0; pattern < 65536; ++pattern) {
            std::array<T, 35> values{};
            for (unsigned j = 0; j < 16; ++j)
                values[j] = (pattern & (1u << j)) ? static_cast<T>(j + 1) : mode;
            values[32] = static_cast<T>(-32768);
            size_t i = 0, block_end = 0;
            uint32_t mask = 0;
            while (i < values.size()) {
                size_t expected = 0;
                while (i + expected < values.size() && values[i + expected] == mode) ++expected;
                const size_t run = E::rle_scan_run_cached(
                    values.data(), i, values.size(), mode, vm, block_end, mask);
                check(run == expected, "AVX2 cached RLE mask");
                i += run;
                if (i < values.size()) ++i;
            }
        }
        const auto &lut = E::o1_lut();
        std::vector<T> codes(65537);
        for (size_t i = 0; i < codes.size(); ++i)
            codes[i] = static_cast<T>(static_cast<uint16_t>(i * 40503u));
        std::array<uint16_t, 256> indices{};
        for (size_t begin = 1; begin < codes.size(); begin += indices.size()) {
            const size_t count = std::min(indices.size(), codes.size() - begin);
            E::o1_prepare_indices(codes.data() + begin, count, indices.data(), lut);
            for (size_t j = 0; j < count; ++j)
                check(indices[j] == 256 * lut.ctx[static_cast<uint16_t>(codes[begin + j - 1])] +
                                        lut.sym256[static_cast<uint16_t>(codes[begin + j])],
                      "AVX2 tANS indices");
        }
        for (size_t count = 1; count <= indices.size(); ++count) {
            E::o1_prepare_indices(codes.data() + 1, count, indices.data(), lut);
            for (size_t j = 0; j < count; ++j)
                check(indices[j] == 256 * lut.ctx[static_cast<uint16_t>(codes[j])] +
                                        lut.sym256[static_cast<uint16_t>(codes[j + 1])],
                      "AVX2 tANS tile tail");
        }
    }
#endif
};
}

namespace {
using Access = Falcons::RleFseEncoderV11TestAccess;

struct Options {
    int method = -1;
    bool delta = false;
    bool simd = true;
    bool vector_api = false;
    bool retain = false;
    int expect_method = -1;
    int expect_delta = -1;
    int expect_overflow = -1;
    int loaded_mode = -1;
    int expect_value_bits = -1;
};

template <class T>
void round_trip(Falcons::RleFseEncoder<T> &encoder, const std::vector<T> &original,
                Options options = {}) {
    std::vector<T> input = original;
    if (options.loaded_mode >= 0) {
        std::array<Falcons::uchar, 5> model{};
        auto *p = model.data();
        Falcons::write(uint8_t{0}, p);
        Falcons::write(uint16_t{0}, p);
        Falcons::write(static_cast<uint16_t>(options.loaded_mode), p);
        const Falcons::uchar *read_model = model.data();
        size_t remaining = model.size();
        encoder.load(read_model, remaining);
        check(remaining == 0 && read_model == model.data() + model.size(), "Loaded RLE model");
    } else if (options.vector_api) encoder.preprocess_encode(input, 0);
    else encoder.preprocess_encode(input.data(), input.size(), 0);
    if (options.method >= 0) Access::force(encoder, options.method, options.delta, options.simd);
    const size_t capacity = 5 * input.size() + 65536;
    std::vector<Falcons::uchar> bytes(capacity + 65, 0xa5);
    auto *position = bytes.data() + 1;
    encoder.save(position);
    const auto *payload = position;
    const size_t encoded = options.vector_api ? encoder.encode(input, position) :
        encoder.encode(input.data(), input.size(), position);
    check(static_cast<size_t>(position - payload) == encoded, "Encode pointer advancement");
    check(position <= bytes.data() + capacity + 1, "Encode output capacity");
    check(bytes[0] == 0xa5, "Encode prefix guard");
    for (size_t j = capacity + 1; j < bytes.size(); ++j)
        check(bytes[j] == 0xa5, "Encode suffix guard");
    if (options.expect_method >= 0)
        check((bytes[1] & 0x7f) == options.expect_method, "Unexpected entropy method");
    if (options.expect_delta >= 0)
        check(((bytes[1] & 0x80) != 0) == (options.expect_delta != 0), "Unexpected delta flag");
    if (options.expect_value_bits >= 0) {
        const auto *p = payload;
        uint64_t nr, nv, s_rcl, s_vcl, rbp, vbp;
        Falcons::read(nr, p);
        Falcons::read(nv, p);
        Falcons::read(s_rcl, p);
        Falcons::read(s_vcl, p);
        Falcons::read(rbp, p);
        Falcons::read(vbp, p);
        check(nr == input.size() && nv == input.size() && rbp == 0,
              "Loaded mode must produce only value symbols");
        check(vbp == (static_cast<size_t>(options.expect_value_bits) * input.size() + 7) / 8,
              "Loaded mode mantissa bit count");
    }
    if (options.expect_overflow >= 0) {
        const auto *p = payload;
        uint64_t segments, n, k;
        Falcons::read(segments, p);
        Falcons::read(n, p);
        Falcons::read(k, p);
        check(segments == 1 && n == input.size() && k == 4, "tANS test block shape");
        uint64_t stream_bytes = 0;
        for (size_t j = 0; j < k; ++j) {
            uint64_t size;
            Falcons::read(size, p);
            stream_bytes += size;
        }
        p += k * sizeof(uint64_t) + 15 * 256 * sizeof(uint16_t) + stream_bytes;
        check(p < position && *p == options.expect_overflow, "Unexpected overflow mode");
    }
    if (!options.retain) {
        encoder.postprocess_encode();
        check(!Access::workspace(encoder), "postprocess retained scratch");
    }
    const size_t size = static_cast<size_t>(position - (bytes.data() + 1));
    const Falcons::uchar *read_position = bytes.data() + 1;
#ifdef FALCONS_V11_TEST_ZSTD
    Falcons::Lossless_zstd zstd;
    std::vector<Falcons::uchar> frame(ZSTD_compressBound(size) + sizeof(size_t));
    const size_t frame_size = zstd.compress(read_position, size, frame.data(), frame.size());
    Falcons::uchar *raw = nullptr;
    size_t raw_size = 0;
    zstd.decompress(frame.data(), frame_size, raw, raw_size);
    std::unique_ptr<Falcons::uchar, decltype(&std::free)> raw_owner(raw, &std::free);
    check(raw_size == size, "Final Zstd payload size");
    read_position = raw;
#endif
    const auto *read_end = read_position + size;
    Falcons::RleFseEncoder<T> decoder;
    size_t remaining = size;
    decoder.load(read_position, remaining);
    decoder.preprocess_decode();
    std::unique_ptr<T[]> decoded(decoder.decode(read_position, original.size()));
    check(read_position == read_end, "Decode pointer advancement");
    for (size_t j = 0; j < original.size(); ++j)
        check(decoded[j] == original[j], "Entropy round-trip mismatch");
    decoder.postprocess_decode();
    ++round_trips;
}

template <class T>
std::vector<T> values(size_t n, int pattern, uint32_t seed = 0x7654321u) {
    std::vector<T> out(n);
    const int edge[] = {-32768, 32767, -128, 127, 128, -127, -1, 1};
    for (size_t i = 0; i < n; ++i) {
        const uint32_t r = next_random(seed);
        switch (pattern) {
            case 0: out[i] = 0; break;
            case 1: out[i] = static_cast<T>(edge[i % 8]); break;
            case 2: out[i] = i % 71 < 3 ? static_cast<T>(edge[i % 8]) : T{0}; break;
            case 3: out[i] = static_cast<T>(static_cast<uint16_t>(r)); break;
            case 4: {
                int code = static_cast<int>(r % 31) - 15;
                if (i && static_cast<T>(code) == out[i - 1]) code = code == 15 ? -15 : code + 1;
                out[i] = static_cast<T>(code);
                break;
            }
            case 5: out[i] = static_cast<T>(edge[(i / 257) % 8]); break;
            default: throw std::runtime_error("Invalid input pattern");
        }
    }
    return out;
}

template <class T>
void test_loaded_modes() {
    Falcons::RleFseEncoder<T> encoder;
    for (int mode : {0, 32767, 32768, 65535}) {
        for (size_t n : {size_t{0}, size_t{1}, size_t{7}, size_t{8}, size_t{127}, size_t{128},
                         size_t{129}, size_t{255}, size_t{256}, size_t{257}, size_t{4096}, size_t{65537}}) {
            for (bool vector_api : {false, true}) {
                Options o;
                o.loaded_mode = mode;
                o.vector_api = vector_api;
                o.retain = true;
                o.expect_method = 0;
                o.expect_delta = 0;
                if (mode == 65535) o.expect_value_bits = std::is_signed<T>::value ? 17 : 16;
                round_trip(encoder, std::vector<T>(n, std::numeric_limits<T>::min()), o);
            }
        }
        Options o;
        o.loaded_mode = mode;
        round_trip(encoder, values<T>(65537, 1), o);
        std::vector<T> domain(65536);
        for (size_t i = 0; i < domain.size(); ++i) domain[i] = static_cast<T>(static_cast<uint16_t>(i));
        round_trip(encoder, domain, o);
    }
    Options delta;
    delta.loaded_mode = 65535;
    delta.method = 0;
    delta.delta = true;
    round_trip(encoder, values<T>(80003, 5), delta);
}

template <class T>
void test_type(bool large) {
    Access::check_scratch<T>();
#ifdef __AVX2__
    Access::check_avx2<T>();
#endif
    test_loaded_modes<T>();
    Falcons::RleFseEncoder<T> encoder;
    for (size_t n : {size_t{0}, size_t{1}, size_t{2}, size_t{15}, size_t{16}, size_t{17},
                     size_t{31}, size_t{32}, size_t{33}, size_t{63}, size_t{64}, size_t{65},
                     size_t{255}, size_t{256}, size_t{257}, size_t{65535}, size_t{65536}, size_t{65537}}) {
        for (int pattern = 0; pattern <= 5; ++pattern) {
            Options o;
            o.vector_api = (pattern & 1) != 0;
            o.retain = true;
            round_trip(encoder, values<T>(n, pattern), o);
        }
    }
    std::vector<T> domain(65536);
    for (size_t i = 0; i < domain.size(); ++i) domain[i] = static_cast<T>(static_cast<uint16_t>(i));
    round_trip(encoder, domain);

    // All stream-count and flush phases, plus partial 256-index tiles.
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{15}, size_t{16}, size_t{17},
                     size_t{1023}, size_t{1024}, size_t{1025}, size_t{1026}, size_t{1027},
                     size_t{1039}, size_t{1040}, size_t{1041}}) {
        Options o;
        o.method = 1;
        o.retain = true;
        round_trip(encoder, values<T>(n, 4), o);
        round_trip(encoder, std::vector<T>(n, static_cast<T>(128)), o);
    }
    for (size_t n : {size_t{76800}, size_t{76801}, size_t{76802}, size_t{76803},
                     size_t{1048575}, size_t{1048576}, size_t{1048577}}) {
        for (int pattern : {3, 4}) {
            Options o;
            o.retain = true;
            o.expect_method = 1;
            o.expect_delta = 0;
            round_trip(encoder, values<T>(n, pattern), o);
        }
    }
    // Change distributions repeatedly without releasing cached tables between blocks.
    Options forced;
    forced.method = 1;
    forced.retain = true;
    for (int pattern : {0, 1, 4, 0, 4, 3, 1, 4, 4})
        round_trip(encoder, values<T>(80003, pattern), forced);
    forced.expect_overflow = 1;
    round_trip(encoder, std::vector<T>(80003, static_cast<T>(128)), forced);
    forced.expect_overflow = 0;
    round_trip(encoder, std::vector<T>(80003, static_cast<T>(-32768)), forced);
    forced.expect_overflow = -1;
    for (int method : {0, 1}) {
        forced.method = method;
        for (bool simd : {false, true}) {
            forced.simd = simd;
            for (bool delta : {false, true}) {
                forced.delta = delta;
                round_trip(encoder, values<T>(80003, 5), forced);
            }
        }
    }
    Options automatic;
    automatic.expect_method = 1;
    automatic.expect_delta = 1;
    round_trip(encoder, values<T>(80003, 5), automatic);
    Falcons::RleFseEncoder<T> copied(encoder), assigned;
    assigned = copied;
    round_trip(copied, values<T>(80003, 4));
    round_trip(assigned, values<T>(80003, 3));
    if (large) {
        Options o;
        o.expect_method = 1;
        round_trip(encoder, values<T>((size_t{16} << 20) + 13, 4), o);
    }
}

void test_concurrent() {
    std::array<std::exception_ptr, 4> failures{};
    std::array<std::thread, 4> workers;
    for (size_t k = 0; k < workers.size(); ++k) {
        workers[k] = std::thread([k, &failures]() {
            try {
                Falcons::RleFseEncoder<int16_t> a;
                Options o;
                o.retain = true;
                round_trip(a, values<int16_t>(80003 + k, 4, static_cast<uint32_t>(k + 1)), o);
                Falcons::RleFseEncoder<int16_t> b(a);
                round_trip(b, values<int16_t>(80003 + k, 3, static_cast<uint32_t>(k + 5)), o);
                round_trip(a, values<int16_t>(65537 + k, 2), o);
                a.postprocess_encode();
                b.postprocess_encode();
            } catch (...) { failures[k] = std::current_exception(); }
        });
    }
    for (auto &worker : workers) worker.join();
    for (const auto &failure : failures)
        if (failure) std::rethrow_exception(failure);
}
}

int main(int argc, char **argv) {
    try {
#ifdef FALCONS_V11_TEST_ZSTD
        check(std::getenv("FALCONS_NOZSTD") == nullptr, "Unset FALCONS_NOZSTD for Zstd tests");
#endif
        bool large = false;
        for (int i = 1; i < argc; ++i) {
            check(std::string(argv[i]) == "--large", "Only --large is supported");
            large = true;
        }
        Access::check_mantissa();
        test_type<int16_t>(large);
        test_type<uint16_t>(large);
        test_concurrent();
        std::cout << "PASS: " << round_trips.load() << " entropy round trips";
#ifdef __AVX2__
        std::cout << ", AVX2 kernels";
#else
        std::cout << ", scalar kernels";
#endif
#ifdef FALCONS_V11_TEST_ZSTD
        std::cout << ", final Zstd";
#endif
        std::cout << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
