function(build_hdf5_test)
    get_filename_component(test_name ${ARGV0} NAME_WE)
    add_executable(${test_name} ${ARGV})
    target_link_libraries(${test_name} PUBLIC Falcons hdf5falcons)
endfunction(build_hdf5_test)

build_hdf5_test(falconsToHDF5.cpp)
build_hdf5_test(HDF5ToFalcons.cpp)
build_hdf5_test(convertBinToHDF5.cpp)
build_hdf5_test(cdvalueHelper.cpp)
set_target_properties(cdvalueHelper PROPERTIES OUTPUT_NAME print_h5repack_args)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/h5repack.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
