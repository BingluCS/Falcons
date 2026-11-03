#include "Falcons/lossless/Lossless_zstd.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void run_case(size_t n, bool compressible) {
    std::vector<Falcons::uchar> source(n);
    uint32_t rng = 7;
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        source[i] = compressible ? static_cast<Falcons::uchar>(i % 5) : static_cast<Falcons::uchar>(rng >> 24);
    }
    Falcons::Lossless_zstd codec;
    const size_t capacity = sizeof(size_t) + ZSTD_compressBound(n);
    std::vector<Falcons::uchar> encoded(capacity + 1, 0x5a);
    size_t written = 0;
    check(codec.compress_bounded_trial(source.data(), n, encoded.data(), capacity, written),
          "bounded trial rejected sufficient capacity");
    check(written > sizeof(size_t) && written <= capacity, "invalid bounded output size");
    Falcons::uchar *decoded = nullptr;
    size_t decoded_size = 0;
    codec.decompress(encoded.data(), written, decoded, decoded_size);
    check(decoded_size == n && std::equal(source.begin(), source.end(), decoded), "bounded trial roundtrip");
    std::free(decoded);
    std::fill(encoded.begin(), encoded.end(), 0x5a);
    size_t ignored = 0;
    check(!codec.compress_bounded_trial(source.data(), n, encoded.data(), written - 1, ignored),
          "bounded trial accepted an undersized destination");
    check(encoded[written - 1] == 0x5a, "bounded trial wrote past destination");
}

int main() {
    try {
        check(std::getenv("FALCONS_NOZSTD") == nullptr, "FALCONS_NOZSTD must be unset");
        run_case(0, true);
        run_case(1, true);
        run_case(1 << 17, true);
        run_case(1 << 17, false);
        std::puts("PASS bounded Zstd trials");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
