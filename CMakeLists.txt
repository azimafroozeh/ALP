cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
project(ALP)

add_compile_options(-fPIC)

# Options : ------------------------------------------------------------------------------------------------------------
option(ALP_BUILD_LIBRARY "Build ALP Library" ON)
option(ALP_BUILD_TESTING "Build Testing" OFF)
option(ALP_BUILD_BENCHMARKING "Build Benchmarking" OFF)
option(ALP_ENABLE_CLANG_TIDY "Enable clang_tidy on all targets" ON)
option(ALP_ENABLE_VERBOSE_OUTPUT "Enable verbose output" OFF)

#-----------------------------------------------------------------------------------------------------------------------
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
# https://stackoverflow.com/questions/56089330/cmake-creates-lots-of-targets-i-didnt-specify
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
include(CTest)
include(GoogleTest)

# Verbose : ------------------------------------------------------------------------------------------------------------
if (ALP_ENABLE_VERBOSE_OUTPUT)
    message("---------------------------------------------------------------------------------------------------------")
    message("-- ALP: Verbose Enabled:")

    cmake_print_variables(
            CMAKE_SYSTEM_PROCESSOR
            CMAKE_SYSTEM_NAME
            CMAKE_VERSION
            CMAKE_BUILD_TYPE
            CMAKE_CXX_COMPILER
            CMAKE_CXX_COMPILER_VERSION
            CMAKE_CXX_STANDARD
            CMAKE_SYSTEM
            CMAKE_HOST_SYSTEM_NAME
            CMAKE_HOST_SYSTEM_PROCESSOR
            CMAKE_GENERATOR
            CMAKE_BINARY_DIR
            CMAKE_CURRENT_SOURCE_DIR
            CMAKE_LINKER
            CMAKE_CXX_FLAGS
            CMAKE_C_FLAGS
            CMAKE_CROSSCOMPILING
    )
endif ()

# Requirements : -------------------------------------------------------------------------------------------------------
if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message(FATAL_ERROR "Only Clang is supported!")
endif ()

# Clang Tidy : ---------------------------------------------------------------------------------------------------------
if (ALP_BUILD_PUBLICATION)
    set(ALP_ENABLE_CLANG_TIDY OFF)
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ALP_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if (NOT CLANG_TIDY_EXE)
        message(FATAL_ERROR "-- ALP: clang-tidy not found.")
    else ()
        set(ALP_CLANG_TIDY_FLAGS
                "-header-filter=${CMAKE_CURRENT_SOURCE_DIR}/include,${CMAKE_CURRENT_SOURCE_DIR}/data/include;-warnings-as-errors=*"
                CACHE STRING "Flags for clang-tidy")
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE};${ALP_CLANG_TIDY_FLAGS})
    endif ()
endif ()

# CMAKE_CURRENT_SOURCE_DIR: ---------------------------------------------------------------------------------------------
add_compile_definitions(ALP_CMAKE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Src: -----------------------------------------------------------------------------------------------------------------
if (ALP_BUILD_LIBRARY)
    message("---------------------------------------------------------------------------------------------------------")
    message("-- ALP: Build Library:")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

    install(
            EXPORT ALPTargets
            FILE ALPTargets.cmake
            DESTINATION lib/ALP
            NAMESPACE ALP::
    )

    export(
            EXPORT ALPTargets
            FILE "${CMAKE_BINARY_DIR}/ALPTargets.cmake"
            NAMESPACE ALP::
    )


endif ()

# Data: ----------------------------------------------------------------------------------------------------------------
if (ALP_BUILD_TESTING OR ALP_BUILD_PUBLICATION OR ALP_BUILD_BENCHMARKING)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/data/include)
endif ()

# Gtest: ---------------------------------------------------------------------------------------------------------------
if (ALP_BUILD_TESTING OR ALP_BUILD_BENCHMARKING OR ALP_BUILD_PUBLICATION)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.13.0
    )
    FetchContent_MakeAvailable(googletest)

    # Silence clang-tidy warnings from googletest
    set_target_properties(gtest PROPERTIES CXX_CLANG_TIDY "")
    set_target_properties(gtest_main PROPERTIES CXX_CLANG_TIDY "")
    set_target_properties(gmock PROPERTIES CXX_CLANG_TIDY "")
    set_target_properties(gmock_main PROPERTIES CXX_CLANG_TIDY "")

    enable_testing()
endif ()

# Test : ---------------------------------------------------------------------------------------------------------------
if (ALP_BUILD_TESTING)
    message("---------------------------------------------------------------------------------------------------------")
    message("-- ALP: Build Testing:")

    include_directories(${CMAKE_SOURCE_DIR}/test/include)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif ()

# Benchmark : ----------------------------------------------------------------------------------------------------------
if (ALP_BUILD_BENCHMARKING)
    message("---------------------------------------------------------------------------------------------------------")
    message("-- ALP: Build Benchmarking:")

    include_directories(${CMAKE_SOURCE_DIR}/benchmarks/include)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
endif ()
