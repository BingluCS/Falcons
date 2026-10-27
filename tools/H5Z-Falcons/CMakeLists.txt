#Either use:
#find_package(HDF5 1.12.1 REQUIRED)
#or manually set the hdf5 paths:
#set(HDF5_LIBRARIES "/path/to/hdf5-1.12.1/hdf5/lib/libhdf5.so")
#set(HDF5_INCLUDE_DIRS "/path/to/hdf5-1.12.1/hdf5/include")

find_package(HDF5 COMPONENTS C REQUIRED)

#message("HDF5LIB PATH:" ${HDF5_LIBRARIES})
#message("HDF5INCLUDE PATH:" ${HDF5_INCLUDE_DIRS})

add_library(hdf5falcons
        src/H5Z_Falcons.cpp
)

target_link_libraries(hdf5falcons
        PUBLIC Falcons HDF5::HDF5
)

# The compression happens inside this filter TU, so it must be built with the SIMD
# flags for the AVX2/SVE2 code paths in the (header-only) Falcons library to activate.
# Linking Falcons (INTERFACE) already propagates these, but set them explicitly to match
# the standalone falcons binary.
if (ENABLE_AVX2)
    target_compile_options(hdf5falcons PRIVATE -mavx2 -mfma)
endif ()
if (ENABLE_SVE2)
    target_compile_options(hdf5falcons PRIVATE -march=armv8.6-a+sve2)
endif ()

if (${HDF5_IS_PARALLEL})
    find_package(MPI REQUIRED)
    target_link_libraries(hdf5falcons
            PUBLIC MPI::MPI_CXX)
endif ()

target_include_directories(hdf5falcons
        PUBLIC
        ${HDF5_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hdf5_falcons>
)

add_subdirectory(test)

install(TARGETS hdf5falcons EXPORT HDF5Falcons
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hdf5_falcons)
export(TARGETS hdf5falcons FILE HDF5Falcons.cmake)
