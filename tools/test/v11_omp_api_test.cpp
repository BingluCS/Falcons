#include "Falcons/api/falcons.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef _OPENMP
int main() { std::puts("SKIP: OpenMP required"); return 77; }
#else
#include <omp.h>

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void run_case(int compress_threads, int decompress_threads, bool dynamic) {
    omp_set_num_threads(compress_threads);
    omp_set_dynamic(dynamic ? 1 : 0);
    Falcons::Config config(40, 40, 64);
    config.cmprAlgo = Falcons::ALGO_INTERP;
    config.errorBoundMode = Falcons::EB_ABS;
    config.absErrorBound = 0.01;
    config.openmp = true;
    std::vector<float> data(config.num);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>(std::sin(i * 0.071) * 50 + i % 17);
    const std::vector<float> original = data;
    size_t compressed_size = 0;
    std::unique_ptr<char[]> compressed(Falcons_compress(config, data.data(), compressed_size));
    check(omp_get_max_threads() == compress_threads && omp_get_dynamic() == int(dynamic),
          "compression changed caller OpenMP settings");
    Falcons::Config stream_config;
    Falcons_read_config(stream_config, compressed.get(), compressed_size);
    int saved_threads = 1;
    if (compress_threads > 1) {
        check(stream_config.openmp, "compression did not use the OpenMP path");
        const size_t header_size = static_cast<unsigned char>(compressed[0]);
        check(compressed_size - header_size >= sizeof(int), "missing OpenMP thread count");
        std::memcpy(&saved_threads, compressed.get() + header_size, sizeof(saved_threads));
        check(saved_threads == compress_threads, "compression team differs from the requested team");
    } else {
        check(!stream_config.openmp, "single-thread compression did not use the serial path");
    }
    omp_set_num_threads(decompress_threads);
    Falcons::Config recovered_config;
    std::unique_ptr<float[]> recovered(Falcons_decompress<float>(recovered_config, compressed.get(), compressed_size));
    check(omp_get_max_threads() == decompress_threads && omp_get_dynamic() == int(dynamic),
          "decompression changed caller OpenMP settings");
    for (size_t i = 0; i < data.size(); ++i)
        check(std::abs(double(original[i]) - recovered[i]) <= config.absErrorBound + 1e-6,
              "parallel API reconstruction exceeds error bound");
    std::printf("PASS OMP API compress=%d streams=%d decompress_default=%d dynamic=%d\n",
                compress_threads, saved_threads, decompress_threads, int(dynamic));
}

static void run_high_escape() {
    omp_set_num_threads(2);
    omp_set_dynamic(0);
    Falcons::Config config((size_t(1) << 21) + (size_t(1) << 16));
    config.cmprAlgo = Falcons::ALGO_INTERP;
    config.errorBoundMode = Falcons::EB_ABS;
    config.absErrorBound = 0.01;
    config.interpAnchorStride = 4096;
    config.openmp = true;
    std::vector<float> data(config.num);
    uint32_t rng = 7;
    for (float &value : data) {
        rng = rng * 1664525u + 1013904223u;
        value = static_cast<float>(static_cast<int>(rng >> 16) - 32768) * 0.005f;
    }
    const auto original = data;
    size_t compressed_size = 0;
    std::unique_ptr<char[]> compressed(Falcons_compress(config, data.data(), compressed_size));
    Falcons::Config recovered_config;
    std::unique_ptr<float[]> recovered(Falcons_decompress<float>(recovered_config, compressed.get(), compressed_size));
    check(recovered_config.cmprAlgo == Falcons::ALGO_INTERP && recovered_config.openmp,
          "high-escape test did not exercise parallel interpolation");
    for (size_t i = 0; i < data.size(); ++i)
        check(std::abs(double(original[i]) - recovered[i]) <= config.absErrorBound + 1e-6,
              "high-escape parallel reconstruction exceeds error bound");
    std::puts("PASS high-escape OMP API");
}

int main() {
    try {
        run_case(32, 1, true);
        run_case(1, 8, false);
        run_high_escape();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
#endif
