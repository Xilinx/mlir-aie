# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Builds the bundled XRT-free, Boost-free `hrx-xclbinutil` and installs it as
# `xclbinutil` next to aiecc, so aiecc's findAieTool("xclbinutil") resolves to
# it with no system XRT install. The source is a TRIMMED copy of XRT's
# xclbinutil (NPU-minimal: only the section handlers an NPU xclbin needs) with
# Boost replaced by util/hrx_util.{h,cpp}; it is vendored as the
# third_party/hrx-xclbinutil submodule.
#
# Enabled by -DAIE_BUILD_HRXXCLBINUTIL=ON (see the option in the top-level
# CMakeLists.txt). Requires: git submodule update --init third_party/hrx-xclbinutil

# Root of the vendored hrx-xclbinutil checkout. Overridable so packagers/tests
# can point at a checkout elsewhere.
set(HRXXCLBINUTIL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/hrx-xclbinutil"
    CACHE PATH "Path to the hrx-xclbinutil source tree")

set(_hx "${HRXXCLBINUTIL_SOURCE_DIR}")
set(_xclbinutil_source_dir "${_hx}/vendor/xclbinutil")
set(_util "${_hx}/util")

if(NOT EXISTS "${_xclbinutil_source_dir}")
  message(FATAL_ERROR
    "AIE_BUILD_HRXXCLBINUTIL=ON but ${_xclbinutil_source_dir} is missing.\n"
    "Initialize the submodule: git submodule update --init third_party/hrx-xclbinutil")
endif()

find_package(Threads REQUIRED)

# transformcdo (aie-pdi-transform) — vendored, C, no external deps. Builds into
# our binary dir so the submodule source tree stays clean.
add_subdirectory("${_xclbinutil_source_dir}/aie-pdi-transform"
                 "${CMAKE_CURRENT_BINARY_DIR}/aie-pdi-transform")

# version.h — generated; pinned to the vendored XRT (2.18.0). Emitted into the
# build tree (not the submodule source, which .gitignores it) and placed first
# on the include path so it always wins.
set(XRT_VERSION_MAJOR 2)
set(XRT_VERSION_MINOR 18)
set(XRT_VERSION_PATCH 0)
set(XRT_VERSION_STRING ${XRT_VERSION_MAJOR}.${XRT_VERSION_MINOR}.${XRT_VERSION_PATCH})
set(XRT_HEAD_COMMITS 0xDEADBEEF)
set(XRT_BRANCH_COMMITS 0xDEADFACE)
set(XRT_BRANCH "mlir-aie-hrx")
set(XRT_HASH "")
set(XRT_HASH_DATE "")
set(XRT_DATE_RFC "")
set(XRT_DATE "")
set(XRT_MODIFIED_FILES "")
configure_file("${_hx}/vendor/version.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h" @ONLY)

# The trimmed vendor/ dir contains exactly the NPU-minimal source set.
file(GLOB _xclbinutil_srcs "${_xclbinutil_source_dir}/*.cxx")

add_executable(hrx-xclbinutil ${_xclbinutil_srcs} "${_util}/hrx_util.cpp")
# aiecc invokes a bare `xclbinutil`; emit and install under that name.
set_target_properties(hrx-xclbinutil PROPERTIES OUTPUT_NAME xclbinutil)
set(THREADS_PREFER_PTHREAD_FLAG ON)
# ws2_32: on Windows ntohl/htonl/ntohs/htons come from winsock2; on POSIX libc.
target_link_libraries(hrx-xclbinutil
                      PRIVATE
                      Threads::Threads
                      $<BUILD_LOCAL_INTERFACE:transformcdo>
                      $<$<PLATFORM_ID:Windows>:ws2_32>)
target_include_directories(hrx-xclbinutil
                           PRIVATE
                                   "${CMAKE_CURRENT_BINARY_DIR}/generated"
                                   "${_util}"
                                   "${_hx}/vendor/include"
                                   "${_xclbinutil_source_dir}")
# -w: the vendored XRT sources are carried verbatim and are not warning-clean
# under mlir-aie's stricter flags; silence them rather than fork the sources.
target_compile_options(hrx-xclbinutil
                       PRIVATE
                       $<$<PLATFORM_ID:Linux>:-fexceptions -frtti -w>
                       $<$<PLATFORM_ID:Windows>:/EHsc /GR /w>)

# Install alongside aiecc/bootgen (default RUNTIME dest is bin) so aiecc finds
# `xclbinutil` in its own executable directory.
install(TARGETS hrx-xclbinutil RUNTIME DESTINATION bin)
