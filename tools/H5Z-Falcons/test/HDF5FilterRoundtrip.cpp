#include "H5Z_Falcons.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
static void roundtrip(hid_t datatype, Falcons::EB mode) {
    const hsize_t dims[2] = {64, 64};
    std::vector<T> input(dims[0] * dims[1]), output(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<T>(std::sin(i * 0.03) + (i % 37) * 0.01);

    Falcons::Config conf(dims[0], dims[1]);
    conf.errorBoundMode = mode;
    conf.absErrorBound = 0.001;
    conf.relErrorBound = 0.001;

    hid_t access = H5Pcreate(H5P_FILE_ACCESS);
    check(access >= 0 && H5Pset_fapl_core(access, 65536, 0) >= 0, "file access");
    hid_t file = H5Fcreate("falcons-filter-test", H5F_ACC_TRUNC, H5P_DEFAULT, access);
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    check(file >= 0 && space >= 0 && dcpl >= 0, "HDF5 setup");
    check(H5Pset_chunk(dcpl, 2, dims) >= 0 && set_Falcons_conf_to_H5(dcpl, conf) > 0, "filter setup");
    hid_t dataset = H5Dcreate2(file, "field", datatype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    check(dataset >= 0, "create dataset");
    check(H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data()) >= 0, "write dataset");
    check(H5Dclose(dataset) >= 0, "close dataset");
    dataset = H5Dopen2(file, "field", H5P_DEFAULT);
    check(dataset >= 0 && H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, output.data()) >= 0,
          "read dataset");
    const auto [minimum, maximum] = std::minmax_element(input.begin(), input.end());
    const double bound = mode == Falcons::EB_REL
        ? (static_cast<double>(*maximum) - static_cast<double>(*minimum)) * conf.relErrorBound
        : conf.absErrorBound;
    for (size_t i = 0; i < output.size(); ++i) {
        check(std::isfinite(output[i]), "non-finite output");
        const double slack = 4 * std::numeric_limits<T>::epsilon() *
                             std::max(1.0, std::abs(static_cast<double>(input[i])));
        check(std::abs(static_cast<double>(output[i]) - input[i]) <= bound + slack,
              "HDF5 reconstruction exceeds error bound");
    }

    H5Dclose(dataset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(access);
}

int main() {
    try {
        check(H5Zregister(H5PLget_plugin_info()) >= 0, "register filter");
        roundtrip<float>(H5T_NATIVE_FLOAT, Falcons::EB_ABS);
        roundtrip<float>(H5T_NATIVE_FLOAT, Falcons::EB_REL);
        roundtrip<double>(H5T_NATIVE_DOUBLE, Falcons::EB_ABS);
        std::puts("PASS HDF5 filter float/double roundtrips");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL HDF5 filter: %s\n", e.what());
        return 1;
    }
}
