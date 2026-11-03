#include "Falcons/encoder/RleFseEncoder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__linux__) && SIZE_MAX > UINT32_MAX
#include <sys/mman.h>
#endif

using Encoder = Falcons::RleFseEncoder<int16_t>;
static constexpr size_t block_size = size_t(1) << 16;
static constexpr size_t header_size = 5;

static void check(bool condition, const char *what) {
    if (!condition) throw std::runtime_error(what);
}

static void force_byte_split(Encoder &encoder) {
    const uint8_t header[header_size] = {2, 0, 0, 0, 0};
    const uint8_t *in = header;
    size_t remaining = sizeof(header);
    encoder.load(in, remaining);
    check(remaining == 0, "forced header length");
}

static std::vector<uint8_t> encode(std::vector<int16_t> input, int expected_flags, bool force = false) {
    Encoder encoder;
    if (force) force_byte_split(encoder);
    else encoder.preprocess_encode(input.data(), input.size(), 0);
    std::vector<uint8_t> bytes(input.size() * 8 + (size_t(1) << 20));
    uint8_t *out = bytes.data();
    encoder.save(out);
    check(out == bytes.data() + header_size && bytes[0] == expected_flags, "unexpected encoder flags");
    const size_t written = encoder.encode(input.data(), input.size(), out);
    check(out == bytes.data() + header_size + written, "encoded size mismatch");
    encoder.postprocess_encode();
    bytes.resize(header_size + written);
    return bytes;
}

static void decode_check(const std::vector<uint8_t> &bytes, const std::vector<int16_t> &expected) {
    const uint8_t *in = bytes.data();
    size_t remaining = bytes.size();
    Encoder decoder;
    decoder.load(in, remaining);
    std::unique_ptr<int16_t[]> decoded(decoder.decode(in, expected.size()));
    check(in == bytes.data() + bytes.size(), "encoded bytes not consumed");
    check(std::equal(expected.begin(), expected.end(), decoded.get()), "encoder roundtrip");
    if ((bytes[0] & 0x7f) == 2)
        for (size_t i = 0; i < 32; ++i) check(decoded[expected.size() + i] == 0, "padding not zero");
}

static void check_rejected(const std::vector<uint8_t> &bytes, size_t count) {
    const uint8_t *in = bytes.data();
    size_t remaining = bytes.size();
    Encoder decoder;
    bool threw = false;
    try {
        decoder.load(in, remaining);
        std::unique_ptr<int16_t[]> decoded(decoder.decode(in, count));
    } catch (const std::runtime_error &) { threw = true; }
    check(threw, "malformed byte split accepted");
}

static uint32_t next_random(uint32_t &state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

static int16_t from_bytes(uint8_t low, uint8_t high) {
    const uint16_t zz = static_cast<uint16_t>(low | (unsigned(high) << 8));
    return static_cast<int16_t>(static_cast<uint16_t>((zz >> 1) ^ (0u - (zz & 1u))));
}

static size_t next_stream(const std::vector<uint8_t> &bytes, size_t pos, size_t n, uint8_t mode) {
    check(pos < bytes.size() && bytes[pos++] == mode, "unexpected byte stream mode");
    if (mode == 0) return pos + n;
    if (mode == 2) return pos + 1;
    check(bytes.size() - pos >= sizeof(uint32_t), "missing FSE size");
    uint32_t length;
    std::memcpy(&length, bytes.data() + pos, sizeof(length));
    pos += sizeof(length);
    check(length <= bytes.size() - pos, "invalid FSE size");
    // A 64K skewed block uses tableLog=11, exercising the 32-bit flush boundary.
    check(length > 8 && (bytes[pos + 8] & 15u) == 11u - FSE_MIN_TABLELOG, "FSE tableLog");
    return pos + length;
}

static void test_stream_modes() {
    uint32_t rng = 1;
    for (uint8_t mode = 0; mode < 3; ++mode) {
        std::vector<int16_t> input(block_size);
        for (size_t i = 0; i < input.size(); ++i) {
            const uint32_t a = next_random(rng), b = next_random(rng);
            const bool rare = i % 32 < 4;
            const uint8_t low = mode == 0 ? uint8_t(a) : mode == 1 ? uint8_t(rare ? 1 + a % 255 : 0) : 73;
            const uint8_t high = mode == 0 ? uint8_t(b) : mode == 1 ? uint8_t(rare ? 2 + b % 254 : 1) : 42;
            input[i] = from_bytes(low, high);
        }
        const auto bytes = encode(input, 2, true);
        size_t pos = next_stream(bytes, header_size, input.size(), mode);
        pos = next_stream(bytes, pos, input.size(), mode);
        check(pos == bytes.size(), "extra byte stream data");
        decode_check(bytes, input);

        auto bad = bytes;
        bad.pop_back();
        check_rejected(bad, input.size());
        bad = bytes;
        bad[header_size] = 3;
        check_rejected(bad, input.size());
        if (mode == 1) {
            bad = bytes;
            std::memset(bad.data() + header_size + 1, 0xff, sizeof(uint32_t));
            check_rejected(bad, input.size());
            bad = bytes;
            bad[header_size + 1 + sizeof(uint32_t)] = 0;
            bad[header_size + 2 + sizeof(uint32_t)] = 0;
            check_rejected(bad, input.size());
        }
    }
    std::puts("PASS FSE/RAW/RLE modes and malformed streams");
}

static void test_tails() {
    uint32_t rng = 13;
    for (size_t n : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(4),
                     size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33),
                     block_size - 1, block_size, block_size + 1, 2 * block_size + 19}) {
        std::vector<int16_t> input(n);
        for (auto &value : input) value = static_cast<int16_t>(next_random(rng));
        if (n >= 4) { input[0] = -32768; input[1] = 32767; input[2] = -1; input[3] = 0; }
        const auto bytes = encode(input, 2, true);
        const size_t blocks = n / block_size + (n % block_size != 0);
        check(bytes.size() - header_size <= 2 * n + 2 * blocks, "byte split bound");
        decode_check(bytes, input);
        if (n) {
            auto truncated = bytes;
            truncated.pop_back();
            check_rejected(truncated, n);
        }
    }
    std::puts("PASS int16 extremes, SIMD tails and block boundaries");
}

static void test_fse_tails() {
    for (size_t n : {size_t(1023), size_t(1024), size_t(1025), size_t(1026), size_t(1027),
                     block_size - 3, block_size - 2, block_size - 1,
                     block_size + 1, block_size + 2, block_size + 3}) {
        std::vector<int16_t> input(n);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t low = i % 8 ? 0 : static_cast<uint8_t>(1 + i % 251);
            const uint8_t high = i % 8 ? 1 : static_cast<uint8_t>(2 + (i / 8) % 254);
            input[i] = from_bytes(low, high);
        }
        const auto bytes = encode(input, 2, true);
        check(bytes[header_size] == 1, "expected compressed FSE block");
        decode_check(bytes, input);
    }
    std::puts("PASS FSE stream remainders and compressed block boundaries");
}

static unsigned mode_of(const std::vector<int16_t> &input) {
    Encoder encoder;
    encoder.preprocess_encode(input.data(), input.size(), 0);
    uint8_t header[header_size], *out = header;
    encoder.save(out);
    check(out == header + sizeof(header), "header size");
    return header[0];
}

static void roundtrip(const std::vector<int16_t> &input, unsigned expected_flags) {
    decode_check(encode(input, static_cast<int>(expected_flags)), input);
}

static void test_gate() {
    const size_t n = size_t(1) << 20;
    std::vector<int16_t> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = i & 1 ? 128 : -127;
    check(mode_of(x) == 2, "exactly 50 percent must select byte split");
    roundtrip(x, 2);
    x[1] = -127;
    check(mode_of(x) == 1, "below 50 percent must retain tANS");
    x[1] = 128; x[0] = 128;
    check(mode_of(x) == 2, "above 50 percent must select byte split");
    x[0] = -127;
    x.resize(n - 1);
    check(mode_of(x) == 2, "high ESC below 1Mi must select byte split");
    x.resize(76800);
    check(mode_of(x) == 2, "high ESC at RLE boundary must select byte split");
    roundtrip(x, 2);
    x.resize(76799);
    check(mode_of(x) == 0, "below RLE boundary must retain RLE");
    x.resize(n);
    for (size_t i = 0; i < n; ++i) x[i] = i & 1 ? -128 : 127;
    check(mode_of(x) == 2, "-128 must count as an escape");

    x.resize(n + 19);
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<int16_t>(1000 + (i / 2) % 8);
    check(mode_of(x) == 0x82, "high raw ESC must select byte split with low-ESC delta");
    roundtrip(x, 0x82);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 8 < 2 || (i & 1)) ? -127 : 127;
    check(mode_of(x) == 0x81, "low raw ESC must retain tANS with high-ESC delta");
    roundtrip(x, 0x81);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (i % 4 < 2) ? 1000 : (i % 4 == 2 ? -1000 : 5000);
    check(mode_of(x) == 0x82, "high raw ESC with delta");
    roundtrip(x, 0x82);
    std::fill(x.begin(), x.end(), int16_t(0));
    check(mode_of(x) == 0, "RLE must stay RLE");
    roundtrip(x, 0);
    std::puts("PASS raw ESC threshold, delta, tANS and RLE selection");
}

static void test_overflow() {
    Encoder encoder;
    force_byte_split(encoder);
    const int16_t input = 0;
    uint8_t storage[16], *out = storage;
    bool threw = false;
    try { encoder.encode(&input, (std::numeric_limits<size_t>::max)(), out); }
    catch (const std::length_error &) { threw = true; }
    check(threw && out == storage, "overflow bound accepted");
    std::puts("PASS byte split size overflow");
}

static void test_parallel() {
#ifdef _OPENMP
    int failures = 0, threads = 0;
    const int dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(2)
    {
        #pragma omp single
        { threads = omp_get_num_threads(); }
        #pragma omp for reduction(+:failures)
        for (int tid = 0; tid < 2; ++tid) {
            try {
                std::vector<int16_t> x((size_t(1) << 20) + 17);
                for (size_t i = 0; i < x.size(); ++i)
                    x[i] = tid ? static_cast<int16_t>(1000 + (i / 2) % 8) : (i & 1 ? 128 : -127);
                roundtrip(x, tid ? 0x82 : 2);
            } catch (...) { ++failures; }
        }
    }
    omp_set_dynamic(dynamic);
    check(!failures, "parallel independent workspaces");
    if (threads < 2) std::puts("SKIP parallel execution: OpenMP limited to one thread");
    else std::puts("PASS two independent high-ESC OpenMP encoders");
#else
    std::puts("SKIP parallel execution: OpenMP unavailable");
#endif
}

#if defined(__linux__) && SIZE_MAX > UINT32_MAX
static void test_over_4g() {
    const size_t n = (size_t(1) << 31) + 17, in_bytes = n * sizeof(int16_t);
    const size_t blocks = n / block_size + (n % block_size != 0);
    void *src = mmap(nullptr, in_bytes, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    check(src != MAP_FAILED, "large sparse mmap failed");
    std::vector<uint8_t> bytes(header_size + 4 * blocks);
    uint8_t *out = bytes.data();
    Encoder encoder;
    force_byte_split(encoder);
    encoder.save(out);
    const size_t written = encoder.encode(static_cast<const int16_t *>(src), n, out);
    check(written == 4 * blocks && out == bytes.data() + bytes.size(), "large encoded size");
    encoder.postprocess_encode();
    const uint8_t *in = bytes.data();
    size_t remaining = bytes.size();
    Encoder decoder;
    decoder.load(in, remaining);
    for (size_t off = 0; off < n;) {
        const size_t len = (std::min)(block_size, n - off);
        std::unique_ptr<int16_t[]> decoded(decoder.decode(in, len));
        check(std::all_of(decoded.get(), decoded.get() + len, [](int16_t x) { return x == 0; }),
              "large stream mismatch");
        off += len;
    }
    check(in == out, "large stream trailing bytes");
    munmap(src, in_bytes);
    std::printf("PASS sparse >4GiB input, %zu elements\n", n);
}
#endif

int main() {
    try {
        test_stream_modes(); test_tails(); test_fse_tails(); test_gate(); test_overflow(); test_parallel();
#if defined(__linux__) && SIZE_MAX > UINT32_MAX
        if (std::getenv("FALCONS_TEST_OVER_4G")) test_over_4g();
#endif
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
