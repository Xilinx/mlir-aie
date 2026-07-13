# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(BOOTGEN_SOURCE ${PROJECT_SOURCE_DIR}/third_party/bootgen)

# Upstream (xilinx_v2026.1) splits sources into per-architecture <dir>/src
# trees; glob each one, matching upstream's own Makefile.
set(bootgen_src_dirs
  common
  spartanup
  versal
  versal_2ve_2vm
  zynq
  zynqmp
  utils)
set(libsources)
foreach(dir ${bootgen_src_dirs})
  file(GLOB dir_sources CONFIGURE_DEPENDS
    ${BOOTGEN_SOURCE}/${dir}/src/*.cpp ${BOOTGEN_SOURCE}/${dir}/src/*.c)
  list(APPEND libsources ${dir_sources})
endforeach()
# main.cpp holds the CLI entry point and is only used by the bootgen
# executable below, not by bootgen-lib.
list(REMOVE_ITEM libsources ${BOOTGEN_SOURCE}/common/src/main.cpp)
# Pregenerated bison/flex grammar sources.
list(APPEND libsources
  ${BOOTGEN_SOURCE}/bisonflex/bif.tab.cpp
  ${BOOTGEN_SOURCE}/bisonflex/bif.yy.cpp
  ${BOOTGEN_SOURCE}/bisonflex/cmdoptions.tab.cpp
  ${BOOTGEN_SOURCE}/bisonflex/cmdoptions.yy.cpp
  ${BOOTGEN_SOURCE}/bisonflex/reginit.tab.cpp
  ${BOOTGEN_SOURCE}/bisonflex/reginit.yy.cpp)

set(bootgen_include_dirs
  ${BOOTGEN_SOURCE}/bisonflex
  ${BOOTGEN_SOURCE}/common/include
  ${BOOTGEN_SOURCE}/spartanup/include
  ${BOOTGEN_SOURCE}/versal/include
  ${BOOTGEN_SOURCE}/versal_2ve_2vm/include
  ${BOOTGEN_SOURCE}/zynq/include
  ${BOOTGEN_SOURCE}/zynqmp/include
  ${BOOTGEN_SOURCE}/utils/include
  ${BOOTGEN_SOURCE}/lms-hash-sigs
  ${BOOTGEN_SOURCE}/win_include)

# malloc.h is deprecated and should not be used
# https://stackoverflow.com/a/56463133
# If you want to use malloc, then include stdlib.h
file(READ ${BOOTGEN_SOURCE}/utils/src/cdo-npi.c FILE_CONTENTS)
string(REPLACE "#include <malloc.h>" "" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${BOOTGEN_SOURCE}/utils/src/cdo-npi.c "${FILE_CONTENTS}")

file(READ ${BOOTGEN_SOURCE}/utils/src/cdo-alloc.c FILE_CONTENTS)
string(REPLACE "#include <malloc.h>" "" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${BOOTGEN_SOURCE}/utils/src/cdo-alloc.c "${FILE_CONTENTS}")


find_package(OpenSSL REQUIRED)
if (OPENSSL_FOUND)
  message (STATUS "OpenSSL found")
  message (STATUS "OpenSSL include directories:" ${OPENSSL_INCLUDE_DIR})
else()
  message (FATAL_ERROR "OpenSSL Not found.")
endif()

find_package(Threads REQUIRED)

# since we explicitly link OpenSSL::applink
file(READ ${BOOTGEN_SOURCE}/common/src/main.cpp FILE_CONTENTS)
string(REPLACE "#include \"openssl/ms/applink.c\"" "" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${BOOTGEN_SOURCE}/common/src/main.cpp "${FILE_CONTENTS}")

# Vendored LMS/HSS post-quantum signature library upstream added for newer
# secure-boot support; built as a static lib instead of shelling out to its
# own Makefile.
#
# Only the pthread threading backend is vendored (upstream's Makefile is
# Linux-only). MSVC has no <pthread.h>, so fall back to our own
# single-threaded implementation of the same API (hss_thread.h allows a
# NULL/0 collection as a valid non-threaded mode).
set(lms_hash_sigs_dir ${BOOTGEN_SOURCE}/lms-hash-sigs)
if(CMAKE_USE_PTHREADS_INIT)
  set(bootgen_hss_thread_source ${lms_hash_sigs_dir}/hss_thread_pthread.c)
else()
  set(bootgen_hss_thread_source ${CMAKE_CURRENT_SOURCE_DIR}/hss_thread_single.c)
endif()
add_library(bootgen-lms-hash-sigs STATIC
  ${lms_hash_sigs_dir}/endian.c
  ${lms_hash_sigs_dir}/fips202.c
  ${lms_hash_sigs_dir}/hss.c
  ${lms_hash_sigs_dir}/hss_alloc.c
  ${lms_hash_sigs_dir}/hss_aux.c
  ${lms_hash_sigs_dir}/hss_common.c
  ${lms_hash_sigs_dir}/hss_compute.c
  ${lms_hash_sigs_dir}/hss_derive.c
  ${lms_hash_sigs_dir}/hss_generate.c
  ${lms_hash_sigs_dir}/hss_keygen.c
  ${lms_hash_sigs_dir}/hss_param.c
  ${lms_hash_sigs_dir}/hss_reserve.c
  ${lms_hash_sigs_dir}/hss_sign.c
  ${lms_hash_sigs_dir}/hss_sign_inc.c
  ${bootgen_hss_thread_source}
  ${lms_hash_sigs_dir}/hss_verify.c
  ${lms_hash_sigs_dir}/hss_verify_inc.c
  ${lms_hash_sigs_dir}/hss_zeroize.c
  ${lms_hash_sigs_dir}/lm_common.c
  ${lms_hash_sigs_dir}/lm_ots_common.c
  ${lms_hash_sigs_dir}/lm_ots_sign.c
  ${lms_hash_sigs_dir}/lm_ots_verify.c
  ${lms_hash_sigs_dir}/lm_verify.c
  ${lms_hash_sigs_dir}/lms_hash.c
  ${lms_hash_sigs_dir}/lms_keys.c
  ${lms_hash_sigs_dir}/lms_sign.c
  ${lms_hash_sigs_dir}/sha256.c)
set_target_properties(bootgen-lms-hash-sigs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bootgen-lms-hash-sigs PRIVATE ${OPENSSL_INCLUDE_DIR})
# PUBLIC: static-lib consumers need these transitively on their link line.
target_link_libraries(bootgen-lms-hash-sigs PUBLIC OpenSSL::Crypto Threads::Threads)
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(bootgen-lms-hash-sigs PRIVATE
    -Wno-deprecated-declarations
    -Wno-sign-compare
    -Wno-type-limits
    -Wno-unused)
endif()

add_library(bootgen-lib STATIC ${libsources})
# bootgen must be compiled with PIE/PIC, irrespective of the remainder of
# the project, because  auditwheel repair's patchelf-based RPATH rewrite
# corrupts the non-PIE binary
set_target_properties(bootgen-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(bootgen_warning_ignores
    -Wno-cast-qual
    -Wno-covered-switch-default
    -Wno-date-time
    -Wno-deprecated-copy
    -Wno-deprecated-declarations
    -Wno-deprecated-register
    -Wno-dynamic-class-memaccess
    -Wno-implicit-fallthrough
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-misleading-indentation
    -Wno-non-virtual-dtor
    -Wno-overloaded-virtual
    -Wno-pointer-bool-conversion
    -Wno-register
    -Wno-reorder
    -Wno-sign-compare
    -Wno-suggest-override
    -Wno-tautological-overlap-compare
    -Wno-unused
  )
else()
  set(bootgen_warning_ignores)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_definitions(bootgen-lib PRIVATE YY_NO_UNISTD_H)
  # Enable correct unwind semantics on MSVC
  target_compile_options(bootgen-lib PRIVATE /EHsc)
  # Suppress MSVC warning noise
  target_compile_options(bootgen-lib PRIVATE /wd4996 /wd4840 /wd4005 /wd5033)
endif()
target_compile_options(bootgen-lib PRIVATE ${bootgen_warning_ignores})
target_include_directories(bootgen-lib PRIVATE ${bootgen_include_dirs} ${OPENSSL_INCLUDE_DIR})
# Make C API header available publicly
target_include_directories(bootgen-lib PUBLIC
  $<BUILD_INTERFACE:${BOOTGEN_SOURCE}/common/include>
  $<INSTALL_INTERFACE:include>)
# PUBLIC: same reason as bootgen-lms-hash-sigs above.
target_link_libraries(bootgen-lib PUBLIC bootgen-lms-hash-sigs Threads::Threads)
install(
  TARGETS bootgen-lib
  DESTINATION lib
  EXPORT bootgen-lib)
install(
  FILES ${BOOTGEN_SOURCE}/common/include/bootgen_c_api.h
  DESTINATION include)

add_executable(bootgen ${BOOTGEN_SOURCE}/common/src/main.cpp)
set_target_properties(bootgen PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bootgen PUBLIC ${bootgen_include_dirs} ${OPENSSL_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/include)
target_compile_options(bootgen PRIVATE ${bootgen_warning_ignores})
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(bootgen PRIVATE /EHsc)
  target_compile_options(bootgen PRIVATE /wd4996 /wd4840 /wd4005 /wd5033)
endif()
target_compile_definitions(bootgen PRIVATE OPENSSL_USE_APPLINK)
target_link_libraries(bootgen PRIVATE bootgen-lib OpenSSL::SSL OpenSSL::applink)
install(TARGETS bootgen)

add_library(cdo_driver_mlir_aie STATIC ${BOOTGEN_SOURCE}/cdo-driver/cdo_driver.c)
target_include_directories(
  cdo_driver_mlir_aie PUBLIC $<BUILD_INTERFACE:${BOOTGEN_SOURCE}/cdo-driver>
  $<INSTALL_INTERFACE:include/cdo_driver>)
if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(cdo_driver_mlir_aie PRIVATE -Wno-cast-qual -Wno-sign-compare)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(cdo_driver_mlir_aie PRIVATE /wd4996 /wd4840 /wd4005 /wd5033)
endif()
install(
  TARGETS cdo_driver_mlir_aie
  DESTINATION lib
  EXPORT cdo_driver_mlir_aie)
install(EXPORT cdo_driver_mlir_aie DESTINATION lib/cmake/aie)