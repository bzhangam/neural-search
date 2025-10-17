#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.11)

project(NeuralSearchPlugin_JNI)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros.cmake)

# ---------------------------------- SETUP ----------------------------------
set(TARGET_LIB_VSAG opensearch_neural_vsag)
set(TARGET_LIBS "")  # Libs to be installed

set(CMAKE_CXX_STANDARD 11)

# Set OS specific variables
if (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    set(CMAKE_MACOSX_RPATH 1)
    set(JVM_OS_TYPE darwin)
    set(LIB_EXT .jnilib)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    set(JVM_OS_TYPE linux)
    set(LIB_EXT .so)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
# Set the CXX_COMPILER_VERSION, CMAKE_CXX_FLAGS, JVM_OS_TYPE, prefix and extension for the target libraries that are built.
    set(CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
    set(JVM_OS_TYPE win32)
    set(LIB_EXT .dll)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
else()
    message(FATAL_ERROR "Unable to run on system: ${CMAKE_SYSTEM_NAME}")
endif()

# Set architecture specific variables
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64)
    set(MACH_ARCH arm64)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
    set(MACH_ARCH x64)
endif()
# ----------------------------------------------------------------------------


# ---------------------------------- Neural VSAG Library ----------------------------------
# download and compile vsag
include (FetchContent)
FetchContent_Declare (
  vsag
  GIT_REPOSITORY https://github.com/antgroup/vsag
  GIT_TAG v0.17.0
)

FetchContent_MakeAvailable(vsag)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/vsag-build/antlr4/install/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/vsag-build/openblas/install/lib)

add_library(${TARGET_LIB_VSAG} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/org_opensearch_neuralsearch_jni_NativeVsagService.cpp
)
target_link_libraries (${TARGET_LIB_VSAG} PRIVATE vsag)
target_include_directories(${TARGET_LIB_VSAG} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{JAVA_HOME}/include
    $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
    ${vsag_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/openblas/install/include
)
add_dependencies(${TARGET_LIB_VSAG} vsag)
opensearch_set_common_properties(${TARGET_LIB_VSAG})
list(APPEND TARGET_LIBS ${TARGET_LIB_VSAG})



# ---------------------------------------------------------------------------
