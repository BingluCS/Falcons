#ifndef Falcons_H5Z_Falcons_H
#define Falcons_H5Z_Falcons_H

#define H5Z_FILTER_Falcons 40024

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "Falcons/api/falcons.hpp"
#include "hdf5.h"
#include "H5PLextern.h"

// Export macros for Windows DLL
#ifdef _WIN32
    #ifdef hdf5falcons_EXPORTS
        #define HDF5Falcons_EXPORT __declspec(dllexport)
    #else
        #define HDF5Falcons_EXPORT __declspec(dllimport)
    #endif
#else
    #define HDF5Falcons_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern hid_t H5Z_Falcons_ERRCLASS;
#define ERROR(FNAME)                                                                    \
    do {                                                                                \
        int saved_errno = errno;                                                         \
        fprintf(stderr, #FNAME " failed at line %d, errno=%d (%s)\n", __LINE__, saved_errno, \
                saved_errno ? strerror(saved_errno) : "ok");                           \
        return 1;                                                                       \
    } while (0)

#define H5Z_Falcons_PUSH_AND_GOTO(MAJ, MIN, RET, MSG)                                              \
    do {                                                                                      \
        H5Epush(H5E_DEFAULT, __FILE__, _funcname_, __LINE__, H5Z_Falcons_ERRCLASS, MAJ, MIN, MSG); \
        return RET;                                                                           \
    } while (0)

static herr_t H5Z_falcons_set_local(hid_t dcpl_id, hid_t type_id, hid_t chunk_space_id);

static size_t H5Z_filter_falcons(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes,
                             size_t *buf_size, void **buf);


HDF5Falcons_EXPORT herr_t set_Falcons_conf_to_H5(const hid_t propertyList, Falcons::Config &conf);

HDF5Falcons_EXPORT herr_t get_Falcons_conf_from_H5(const hid_t propertyList, Falcons::Config &conf);

#ifdef __cplusplus
}
#endif

#endif  // Falcons_H5Z_Falcons_H
