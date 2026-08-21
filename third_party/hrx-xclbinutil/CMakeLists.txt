# SPDX-License-Identifier: Apache-2.0
#
# Self-contained, Boost-free, NPU-minimal build of `hrx-xclbinutil`.
#
# Source under ./vendor/ is a TRIMMED copy of iree-amd-aie/third_party/XRT
# (nod-ai/XRT @ a9fdf618): only the 30 .cxx an NPU xclbin needs (MEM_TOPOLOGY,
# AIE_PARTITION, EMBEDDED_METADATA + what --add-kernel/partition touch). Boost is
# replaced by ./util/hrx_util.{h,cpp} (std-backed `hrx::` reimplementation of the
# Boost subset XRT's xclbinutil used). No Boost, no system XRT, no submodule.
#
#   cmake -G Ninja -B bld -S . -DCMAKE_BUILD_TYPE=Release
#   cmake --build bld --target hrx-xclbinutil
#   -> bld/tools/hrx-xclbinutil

cmake_minimum_required(VERSION 3.20)
project(hrx_xclbinutil CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(_vendor ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
set(_xclbinutil_source_dir ${_vendor}/xclbinutil)
set(_util ${CMAKE_CURRENT_SOURCE_DIR}/util)

find_package(Threads REQUIRED)

# transformcdo (aie-pdi-transform) — vendored, C, no deps
add_subdirectory(${_xclbinutil_source_dir}/aie-pdi-transform aie-pdi-transform)

# version.h (generated; pinned to vendored XRT 2.18.0)
set(XRT_VERSION_MAJOR 2)
set(XRT_VERSION_MINOR 18)
set(XRT_VERSION_PATCH 0)
set(XRT_VERSION_STRING ${XRT_VERSION_MAJOR}.${XRT_VERSION_MINOR}.${XRT_VERSION_PATCH})
set(XRT_HEAD_COMMITS 0xDEADBEEF)
set(XRT_BRANCH_COMMITS 0xDEADFACE)
set(XRT_BRANCH "iree-amd-aie-patches")
set(XRT_HASH "")
set(XRT_HASH_DATE "")
set(XRT_DATE_RFC "")
set(XRT_DATE "")
set(XRT_MODIFIED_FILES "")
configure_file(${_vendor}/version.h.in ${_xclbinutil_source_dir}/version.h)

# The trimmed vendor/ dir now contains exactly the NPU-minimal source set.
file(GLOB _xclbinutil_srcs "${_xclbinutil_source_dir}/*.cxx")
set(_util_srcs "${_util}/hrx_util.cpp")

add_executable(hrx-xclbinutil ${_xclbinutil_srcs} ${_util_srcs})
set(THREADS_PREFER_PTHREAD_FLAG ON)
# ws2_32: on Windows ntohl/htonl/ntohs/htons come from winsock2 (DTC.cxx,
# FDTNode.cxx, FDTProperty.cxx, XclBinUtilities.cxx); on POSIX they are in libc.
target_link_libraries(hrx-xclbinutil
                      PRIVATE
                      Threads::Threads
                      $<BUILD_LOCAL_INTERFACE:transformcdo>
                      $<$<PLATFORM_ID:Windows>:ws2_32>)
target_include_directories(hrx-xclbinutil
                           PRIVATE
                                   ${_util}
                                   ${_vendor}/include
                                   ${_xclbinutil_source_dir})
target_compile_options(hrx-xclbinutil
                       PRIVATE
                       $<$<PLATFORM_ID:Linux>:-fexceptions -frtti -w>
                       $<$<PLATFORM_ID:Windows>:/EHsc /GR /w>)
set_target_properties(hrx-xclbinutil
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")

# --- tests (ctest) ---
enable_testing()
add_test(NAME smoke
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test/run_tests.sh $<TARGET_FILE:hrx-xclbinutil>)

# Reaches aie-pdi-transform, which the smoke test does not: the tool's --transform-pdi CLI
# path is #ifndef _WIN32, but the library builds everywhere, so linking it directly gives
# the PDI header structs runtime coverage under MSVC cl too. Golden output was produced by
# the pre-change code and must stay byte-for-byte stable.
add_executable(pdi_transform_test test/pdi_transform_test.c)
target_link_libraries(pdi_transform_test PRIVATE transformcdo)
add_test(NAME pdi_transform
         COMMAND $<TARGET_FILE:pdi_transform_test>
                 ${CMAKE_CURRENT_SOURCE_DIR}/test/data/sample.pdi
                 ${CMAKE_CURRENT_SOURCE_DIR}/test/data/sample_transformed.pdi
                 ${CMAKE_CURRENT_BINARY_DIR}/pdi_transform_test_out.pdi)
