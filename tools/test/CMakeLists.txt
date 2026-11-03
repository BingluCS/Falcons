enable_testing()
# Try to find GTest locally
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found locally; fetching from GitHub...")
    include(FetchContent)
    option(BUILD_GMOCK "Build Google Mock" OFF)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.16.0
    )
    FetchContent_MakeAvailable(googletest)
else()
    message(STATUS "Using locally installed GTest")
endif()

include(CTest)
include(GoogleTest)

# Collect all test source files
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tools/test/modules/*.cpp")

foreach (TEST_SRC IN LISTS TEST_SOURCES)
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    message(STATUS "Test target = ${TEST_NAME}")
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE GTest::gtest_main Falcons)
    # Automatically discover tests for this executable
    # Use PRE_TEST mode to avoid running executable at build time (important for Windows DLL path issues)
    gtest_discover_tests(${TEST_NAME} DISCOVERY_MODE PRE_TEST)
endforeach()

# Standalone regression programs exercise internal buffers and tuning paths.
file(GLOB V11_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/v11_*_test.cpp")
foreach (TEST_SRC IN LISTS V11_TEST_SOURCES)
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE Falcons)
    if(TEST_NAME STREQUAL "v11_entropy_test")
        target_compile_definitions(${TEST_NAME} PRIVATE FALCONS_V11_TEST_ZSTD)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    if(TEST_NAME STREQUAL "v11_omp_test")
        set_tests_properties(${TEST_NAME} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endforeach()
