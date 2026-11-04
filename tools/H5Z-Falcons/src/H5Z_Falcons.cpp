#include "H5Z_Falcons.hpp"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif


hid_t H5Z_Falcons_ERRCLASS = -1;

// filter definition
const H5Z_class2_t H5Z_Falcons[1] = {{
    H5Z_CLASS_T_VERS,                                       /* H5Z_class_t version */
    H5Z_FILTER_Falcons,                                         /* Filter id number */
    1,                                                      /* encoder_present flag (set to true) */
    1,                                                      /* decoder_present flag (set to true) */
    "Falcons compressor/decompressor for floating-point data.", /* Filter name for debugging */
    NULL,                                                   /* The "can apply" callback */
    H5Z_falcons_set_local,                                      /* The "set local" callback */
    static_cast<H5Z_func_t>(H5Z_filter_falcons),                /* The actual filter function */
}};

HDF5Falcons_EXPORT H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_FILTER; }

HDF5Falcons_EXPORT const void* H5PLget_plugin_info(void) { return H5Z_Falcons; }

static herr_t set_Falcons_conf_to_H5_impl(const hid_t propertyList, Falcons::Config& conf,
                                          std::optional<unsigned int> num_threads) {
    static char const* _funcname_ = "set_Falcons_conf_to_H5";

    // save conf into cd_values
    size_t cd_nelmts = (conf.size_est() + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    std::vector<unsigned int> cd_values(cd_nelmts, 0);
    auto buffer = reinterpret_cast<unsigned char*>(cd_values.data());

    auto confSizeReal = conf.save(buffer);
    cd_nelmts = (confSizeReal + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    cd_values.resize(cd_nelmts);
    if (num_threads) cd_values.push_back(*num_threads);
    cd_nelmts = cd_values.size();

    auto falconsfilter = H5Pget_filter_by_id(propertyList, H5Z_FILTER_Falcons, nullptr, nullptr, nullptr, 0, nullptr,
                                            nullptr); // check if filter is set
    if (0 > falconsfilter) {
        // filter not set, set filter. Notice that calling H5Pset_filter twice with the same filter id
        // will cause unexpected errors for decompression
        if (0 > H5Pset_filter(propertyList, H5Z_FILTER_Falcons, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values.data())) {
            H5Z_Falcons_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0, "failed to modify cd_values");
        }
    } else {
        // filter already set, update filter
        if (0 > H5Pmodify_filter(propertyList, H5Z_FILTER_Falcons, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values.data()))
            H5Z_Falcons_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0, "failed to modify cd_values");
    }

    return 1;
}

herr_t set_Falcons_conf_to_H5(const hid_t propertyList, Falcons::Config& conf) {
    return set_Falcons_conf_to_H5_impl(propertyList, conf, std::nullopt);
}

static herr_t get_Falcons_cd_values(const hid_t propertyList, std::vector<unsigned int>& cd_values) {
    static char const* _funcname_ = "get_Falcons_conf_from_H5";

    size_t cd_nelmts = 0;
    if (0 > H5Pget_filter_by_id(propertyList, H5Z_FILTER_Falcons, nullptr, &cd_nelmts, nullptr, 0, nullptr,
                                nullptr))
        H5Z_Falcons_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "unable to get current Falcons cd_values");

    cd_values.resize(cd_nelmts);
    if (cd_nelmts != 0 &&
        0 > H5Pget_filter_by_id(propertyList, H5Z_FILTER_Falcons, nullptr, &cd_nelmts, cd_values.data(), 0, nullptr,
                                nullptr))
        H5Z_Falcons_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "unable to get current Falcons cd_values");
    return 1;
}

static void load_Falcons_conf(const std::vector<unsigned int>& cd_values, Falcons::Config& conf) {
    if (!cd_values.empty()) {
        auto buffer = reinterpret_cast<const unsigned char*>(cd_values.data());
        conf.load(buffer, cd_values.size() * sizeof(unsigned int));
    }
}

static std::optional<unsigned int> get_num_threads(size_t cd_nelmts, const unsigned int* cd_values) {
    if (cd_nelmts == 0) return std::nullopt;
    const auto buffer = reinterpret_cast<const unsigned char*>(cd_values);
    // The optional thread count follows the padded serialized config.
    const size_t conf_words = (static_cast<size_t>(buffer[0]) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    if (cd_nelmts <= conf_words) return std::nullopt;
    const unsigned int num_threads = cd_values[conf_words];
    if (num_threads == 0 || num_threads > static_cast<unsigned int>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid Falcons OpenMP thread count");
    return num_threads;
}

herr_t get_Falcons_conf_from_H5(const hid_t propertyList, Falcons::Config& conf) {
    std::vector<unsigned int> cd_values;
    if (get_Falcons_cd_values(propertyList, cd_values) <= 0) return 0;
    load_Falcons_conf(cd_values, conf);
    return 1;
}

#ifdef _OPENMP
// OpenMP settings belong to the current task and must also be restored on failure.
class ScopedOmpSettings {
public:
    explicit ScopedOmpSettings(std::optional<unsigned int> num_threads) : active(num_threads.has_value()) {
        if (active) {
            threads = omp_get_max_threads();
            dynamic = omp_get_dynamic();
            omp_set_num_threads(static_cast<int>(*num_threads));
            omp_set_dynamic(0);
        }
    }

    ~ScopedOmpSettings() {
        if (active) {
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
        }
    }

private:
    bool active;
    int threads = 0;
    int dynamic = 0;
};
#endif

static herr_t H5Z_falcons_set_local(hid_t dcpl_id, hid_t type_id, hid_t chunk_space_id) {
    printf("start H5Z_falcons_set_local\n");

    // printf("start in H5Z_falcons_set_local, dcpl_id = %d\n", dcpl_id);
    static char const* _funcname_ = "H5Z_falcons_set_local";

    // herr_t ret = H5Zregister(H5Z_Falcons);

    try {
        Falcons::Config conf;
        std::vector<unsigned int> cd_values;
        if (get_Falcons_cd_values(dcpl_id, cd_values) <= 0) return -1;
        load_Falcons_conf(cd_values, conf);
        const auto num_threads = get_num_threads(cd_values.size(), cd_values.data());

        // read datatype and dims from HDF5
        H5T_class_t dclass;
        if (0 > (dclass = H5Tget_class(type_id)))
            H5Z_Falcons_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a datatype");

        size_t dsize;
        if (0 == (dsize = H5Tget_size(type_id)))
            H5Z_Falcons_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "size is smaller than 0!");

        int ndims;
        hsize_t dims_all[H5S_MAX_RANK];
        if (0 > (ndims = H5Sget_simple_extent_dims(chunk_space_id, dims_all, NULL)))
            H5Z_Falcons_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a data space");
        std::vector<size_t> dims(dims_all, dims_all + ndims);
        // update conf with datatype
        conf.dataType = Falcons_FLOAT;
        if (dclass == H5T_FLOAT)
            conf.dataType = dsize == 4 ? Falcons_FLOAT : Falcons_DOUBLE;
        else if (dclass == H5T_INTEGER) {
            H5T_sign_t dsign;
            if (0 > (dsign = H5Tget_sign(type_id)))
                H5Z_Falcons_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "Error in calling H5Tget_sign(type_id)....");
            if (dsign == H5T_SGN_NONE) // unsigned
            {
                switch (dsize) {
                    case 1:
                        conf.dataType = Falcons_UINT8;
                        break;
                    case 2:
                        conf.dataType = Falcons_UINT16;
                        break;
                    case 4:
                        conf.dataType = Falcons_UINT32;
                        break;
                    case 8:
                        conf.dataType = Falcons_UINT64;
                        break;
                }
            } else {
                switch (dsize) {
                    case 1:
                        conf.dataType = Falcons_INT8;
                        break;
                    case 2:
                        conf.dataType = Falcons_INT16;
                        break;
                    case 4:
                        conf.dataType = Falcons_INT32;
                        break;
                    case 8:
                        conf.dataType = Falcons_INT64;
                        break;
                }
            }
        } else {
            H5Z_Falcons_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0, "datatype class must be H5T_FLOAT or H5T_INTEGER");
        }
        // update conf with dims
        conf.setDims(std::begin(dims), std::end(dims));
        //  need to update magic number and data version,
        //  as the config may be from cd_values passed by users
        conf.falconsMagicNumber = Falcons_MAGIC_NUMBER;
        conf.falconsDataVer = versionInt(Falcons_DATA_VER);

        if (set_Falcons_conf_to_H5_impl(dcpl_id, conf, num_threads) <= 0) return -1;
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Falcons HDF5 set_local: %s\n", e.what());
        return -1;
    } catch (...) {
        std::fprintf(stderr, "Falcons HDF5 set_local: unknown failure\n");
        return -1;
    }
}

template <typename T>
size_t process_data(Falcons::Config& conf, void** buf, size_t* buf_size, size_t nbytes, bool is_decompress) {
    const size_t capacity = is_decompress
        ? Falcons::buffer_bounds::mul(conf.num, sizeof(T))
        : Falcons::Falcons_compress_size_bound<T>(conf);
    std::unique_ptr<void, decltype(&std::free)> output(std::malloc(capacity), &std::free);
    if (!output) throw std::bad_alloc();

    size_t output_size;
    if (is_decompress) {
        T* decoded = static_cast<T*>(output.get());
        Falcons_decompress(conf, static_cast<char*>(*buf), nbytes, decoded, capacity / sizeof(T));
        output_size = Falcons::buffer_bounds::mul(conf.num, sizeof(T));
    } else {
        output_size = Falcons_compress(conf, static_cast<T*>(*buf), static_cast<char*>(output.get()), capacity);
    }
    std::free(*buf);
    *buf = output.release();
    *buf_size = capacity;
    return output_size;
}

/**
 * https://docs.hdfgroup.org/hdf5/v1_14/_f_i_l_t_e_r.html
 * The flags, cd_nelmts, and cd_values are the same as for the H5Pset_filter() function with the additional flag
 * H5Z_FLAG_REVERSE which is set when the filter is called as part of the input pipeline. The input buffer is pointed to
 * by *buf and has a total size of *buf_size bytes but only nbytes are valid data. The filter should perform the
 * transformation in place if possible and return the number of valid bytes or zero for failure. If the transformation
 * cannot be done in place then the filter should allocate a new buffer with malloc() and assign it to *buf, assigning
 * the allocated size of that buffer to *buf_size. The old buffer should be freed by calling free().
 */
static size_t H5Z_filter_falcons(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes,
                             size_t* buf_size, void** buf) {
    printf("start H5Z_filter_falcons\n");

    if (cd_nelmts == 0) // this is special data such as string, which should not be treated as values.
        return nbytes;

    try {
        Falcons::Config conf;
        auto buffer = reinterpret_cast<const unsigned char*>(cd_values);
        conf.load(buffer, cd_nelmts * sizeof(unsigned int));
        if (conf.num < 20) return nbytes;

        const auto num_threads = get_num_threads(cd_nelmts, cd_values);
#ifdef _OPENMP
        ScopedOmpSettings omp_settings(num_threads);
#else
        (void)num_threads;
#endif
        bool is_decompress = flags & H5Z_FLAG_REVERSE;
        switch (conf.dataType) {
        case Falcons_FLOAT:
            return process_data<float>(conf, buf, buf_size, nbytes, is_decompress);
#if (!Falcons_DEBUG_TIMINGS)
        case Falcons_DOUBLE:
            return process_data<double>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_INT8:
            return process_data<int8_t>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_UINT8:
            return process_data<uint8_t>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_INT16:
            return process_data<int16_t>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_UINT16:
            return process_data<uint16_t>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_INT32:
            return process_data<int32_t>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_UINT32:
            return process_data<uint32_t>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_INT64:
            return process_data<int64_t>(conf, buf, buf_size, nbytes, is_decompress);
        case Falcons_UINT64:
            return process_data<uint64_t>(conf, buf, buf_size, nbytes, is_decompress);
#endif
        default:
            std::fprintf(stderr, "Falcons HDF5 filter: unsupported datatype\n");
            return 0;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Falcons HDF5 filter: %s\n", e.what());
        return 0;
    } catch (...) {
        std::fprintf(stderr, "Falcons HDF5 filter: unknown failure\n");
        return 0;
    }
}
