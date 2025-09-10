# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.31)
project(mha)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Configuration variables
set(num_KV_heads 4)

set(S_q 512)
set(heads 8) 
set(S_kv 512)
set(d 64)
set(B_q 64)
set(k 64)
set(B_kv 64)
set(dtype "bf16")

set(verbose 0)
set(trace_size 65536)
set(device_name "npu2")
set(design_name "mha")

set(currentDesignTarget ${design_name})

# Derived variables
set(target_suffix "${heads}_${S_q}_${S_kv}_${d}_${B_q}x${d}x${B_kv}")

set(mm_kernel "mm_${B_q}x${d}x${B_kv}_${dtype}_${dtype}")
set(softmax_kernel "softmax_${B_kv}_${dtype}_${dtype}")
set(mha_kernel "mha_${heads}_${S_q}_${S_kv}_${d}_${B_q}x${d}x${B_kv}_${dtype}")
set(passthrough_kernel "passthrough_${B_q}x${B_kv}_${dtype}")

set(build_dir "${CMAKE_BINARY_DIR}")

if(verbose STREQUAL "1")
    set(verbose_string "--verbose")
else()
    set(verbose_string "")
endif()

# Data type mappings
if(dtype STREQUAL "bf16")
    set(dtype_cpp "std::bfloat16_t")
elseif(dtype STREQUAL "f32")
    set(dtype_cpp "float")
endif()

# Display configuration
message(STATUS "Registering Executable: ${currentDesignTarget}")
message(STATUS "    Heads: ${heads}")
message(STATUS "    Number of KV Heads: ${num_KV_heads}")
message(STATUS "    Sequence Length Q: ${S_q}")
message(STATUS "    Sequence Length KV: ${S_kv}")
message(STATUS "    Embedding Dimension d: ${d}")
message(STATUS "    Tile dimensions: ${B_q}x${d}x${B_kv}")
message(STATUS "    Data type: ${dtype}")
message(STATUS "    Device: ${device_name}")
message(STATUS "    Design: ${design_name}")
message(STATUS "    Verbose mode: ${verbose}")

# Platform detection
find_program(WSL NAMES powershell.exe)

if (NOT WSL)
    set(CMAKE_C_COMPILER gcc-13)
    set(CMAKE_CXX_COMPILER g++-13)
    set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT include")
    set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to XRT libraries")
else()
    set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT include")
    set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to XRT libraries")
endif()

# Environment validation
if(NOT DEFINED ENV{PEANO_INSTALL_DIR})
    message(FATAL_ERROR "PEANO_INSTALL_DIR environment variable not set. Please source utils/env_setup.sh")
endif()

# Find required tools
find_program(python_executable python REQUIRED)
find_program(aiecc_executable aiecc.py REQUIRED)

# Path configuration
set(kernels_dir_aie2 "${CMAKE_SOURCE_DIR}/../../../aie_kernels/aie2")
set(kernels_dir "${CMAKE_SOURCE_DIR}/../../../aie_kernels/aie2p")
set(kernels_generic_dir "${CMAKE_SOURCE_DIR}/../../../aie_kernels/generic")
set(runtime_lib_dir "${CMAKE_SOURCE_DIR}/../../../runtime_lib")
set(test_lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/test_lib")


find_program(aieopt_dir aie-opt)

# Build artifact paths
set(golden_ref_header "${build_dir}/golden_reference.h")
set(mlir_target "${build_dir}/aie_${target_suffix}.mlir")
set(xclbin_target "${build_dir}/final_${target_suffix}.xclbin")
set(insts_target "${build_dir}/insts_${target_suffix}.txt")

set(mm_kernel_obj "${build_dir}/${mm_kernel}.o")
set(mm_kernel_obj_rowmaj "${build_dir}/${mm_kernel}_rowmaj.o")
set(softmax_kernel_obj "${build_dir}/${softmax_kernel}.o")
set(mha_kernel_obj "${build_dir}/${mha_kernel}.o")
set(passthrough_kernel_obj "${build_dir}/${passthrough_kernel}.o")

set(kernels_archive "${build_dir}/kernels.a")

# Kernel compilation flags
set(kernel_cc "$ENV{PEANO_INSTALL_DIR}/bin/clang++")
set(warning_flags 
    "-Wno-parentheses"
    "-Wno-attributes"
    "-Wno-macro-redefined"
    "-Wno-empty-body"
    "-Wno-unknown-attributes")
set(kernel_defines
    "-D${dtype}_${dtype}_ONLY"
    "-DDIM_M=${B_q}" 
    "-DDIM_K=${d}" 
    "-DDIM_N=${B_kv}"
    "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"
)

list(APPEND kernel_defines "-DBIT_WIDTH=16")

if(device_name STREQUAL "npu")
    set(kernel_cflags_base 
        "-O2" "-std=c++20" "--target=aie2-none-unknown-elf"
        ${warning_flags} "-DNDEBUG" "-I$ENV{MLIR_AIE_INSTALL_DIR}/include"
    )
elseif(device_name STREQUAL "npu2")
    set(kernel_cflags_base 
        "-O2" "-std=c++20" "--target=aie2p-none-unknown-elf"
        ${warning_flags} "-DNDEBUG" "-I$ENV{MLIR_AIE_INSTALL_DIR}/include"
    )
endif()

set(kernel_cflags ${kernel_cflags_base} ${kernel_defines})

# AIE compiler flags
set(aiecc_flags
    "--alloc-scheme=basic-sequential"
    "--aie-generate-xclbin"
    "--no-compile-host"
    "--no-xchesscc"
    "--no-xbridge"
    "--peano" "$ENV{PEANO_INSTALL_DIR}"
    "--aie-generate-npu-insts"
)

add_custom_command(
    OUTPUT ${golden_ref_header}
    COMMAND ${python_executable} ${CMAKE_CURRENT_SOURCE_DIR}/mha_golden.py
        --heads ${heads} --S_q ${S_q} --S_kv ${S_kv} -d ${d}
        --num_KV_heads ${num_KV_heads}
        --dtype ${dtype}
        --output ${golden_ref_header} ${verbose_string}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mha_golden.py
    COMMENT "Generating PyTorch golden reference"
)

### Kernel compilation ###
add_custom_command(
    OUTPUT ${mm_kernel_obj}
    COMMAND ${kernel_cc} ${kernel_cflags} -DB_COL_MAJ
        -c ${kernels_dir}/mm.cc -o ${mm_kernel_obj}
    DEPENDS ${kernels_dir}/mm.cc
    COMMENT "Compiling AIE kernel: ${mm_kernel} in column-major format"
)

add_custom_command(
    OUTPUT ${mm_kernel_obj_rowmaj}
    COMMAND ${kernel_cc} ${kernel_cflags}
        -c ${kernels_dir}/mm.cc -o ${mm_kernel_obj_rowmaj}
    DEPENDS ${kernels_dir}/mm.cc
    COMMENT "Compiling AIE kernel: ${mm_kernel} in row-major format"
)

add_custom_command(
    OUTPUT ${softmax_kernel_obj}
    COMMAND ${kernel_cc} ${kernel_cflags} 
        -c ${kernels_dir}/softmax.cc -o ${softmax_kernel_obj}
    DEPENDS ${kernels_dir}/softmax.cc
    COMMENT "Compiling AIE kernel: ${softmax_kernel}"
)

add_custom_command(
    OUTPUT ${mha_kernel_obj}
    COMMAND ${kernel_cc} ${kernel_cflags} 
        -c ${kernels_dir}/mha.cc -o ${mha_kernel_obj}
    DEPENDS ${kernels_dir}/mha.cc
    COMMENT "Compiling AIE kernel: ${mha_kernel}"
)

add_custom_command(
    OUTPUT ${passthrough_kernel_obj}
    COMMAND ${kernel_cc} ${kernel_cflags} 
        -c ${kernels_generic_dir}/passThrough.cc -o ${passthrough_kernel_obj}
    DEPENDS ${kernels_generic_dir}/passThrough.cc
    COMMENT "Compiling AIE kernel: ${passthrough_kernel}"
)

add_custom_command(
        OUTPUT ${kernels_archive}
        COMMAND ar rvs ${kernels_archive} ${mm_kernel_obj} ${mm_kernel_obj_rowmaj}_renamed ${softmax_kernel_obj} ${passthrough_kernel_obj} ${mha_kernel_obj} > /dev/null
        DEPENDS ${mm_kernel_obj} ${mm_kernel_obj_rowmaj}_renamed ${softmax_kernel_obj} ${passthrough_kernel_obj} ${mha_kernel_obj}
        COMMENT "Creating kernels archive for NPU2"
    )

# Rename symbols in row-major object file
add_custom_command(
    OUTPUT ${mm_kernel_obj_rowmaj}_renamed
    COMMAND llvm-objcopy-18 --redefine-sym matmul_bf16_bf16=matmul_bf16_bf16_rowmaj
        --redefine-sym matmul_scalar_bf16_bf16=matmul_scalar_bf16_bf16_rowmaj
        --redefine-sym zero_bf16=zero_bf16_rowmaj
        --redefine-sym zero_scalar_bf16=zero_scalar_bf16_rowmaj
        ${mm_kernel_obj_rowmaj} ${mm_kernel_obj_rowmaj}_renamed
    DEPENDS ${mm_kernel_obj_rowmaj}
    COMMENT "Renaming matmul_bf16_bf16 to matmul_bf16_bf16_rowmaj"
)

# MLIR generation
add_custom_command(
    OUTPUT ${mlir_target}
    COMMAND ${python_executable} ${CMAKE_CURRENT_SOURCE_DIR}/${design_name}.py
        --heads ${heads}
        --S_q ${S_q} --S_kv ${S_kv} -d ${d}
        --B_q ${B_q} --B_kv ${B_kv}
        --num_KV_heads ${num_KV_heads}
        --trace_size 0 ${verbose_string} 
        --output_file_path ${mlir_target}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${design_name}.py
    COMMENT "Generating MLIR for ${target_suffix}"
)

# XCLBIN generation
add_custom_command(
    OUTPUT ${xclbin_target} ${insts_target}
    COMMAND ${aiecc_executable} ${aiecc_flags}
        --xclbin-name=final_${target_suffix}.xclbin
        --npu-insts-name=insts_${target_suffix}.txt
        -j 16
        --progress
        ${mlir_target}
    DEPENDS ${mlir_target} ${kernels_archive}
    WORKING_DIRECTORY ${build_dir}
    COMMENT "Generating XCLBIN and instructions for ${target_suffix}"
)

# Host executable
add_executable(${currentDesignTarget}
    ${test_lib_dir}/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

target_compile_definitions(${currentDesignTarget} PUBLIC 
    DISABLE_ABI_CHECK=1
)

target_include_directories (${currentDesignTarget} PUBLIC
    ${XRT_INC_DIR}
    ${test_lib_dir}
    ${runtime_lib_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${build_dir}
    "$ENV{MLIR_AIE_INSTALL_DIR}/include"
)

target_link_directories(${currentDesignTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentDesignTarget} PUBLIC
    xrt_coreutil
)

target_compile_options(${currentDesignTarget} PRIVATE
    -ggdb
)

add_dependencies(${currentDesignTarget} golden_ref xclbin)
add_custom_target(golden_ref DEPENDS ${golden_ref_header})
add_custom_target(xclbin ALL DEPENDS ${xclbin_target} ${insts_target})

# Custom target to run the executable
add_custom_target(run
    COMMAND $<TARGET_FILE:${currentDesignTarget}> -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE --heads ${heads} --S_q ${S_q} --S_kv ${S_kv} -d ${d} -v 2 --warmup 1 --iters 1 --verbosity ${verbose}
    DEPENDS ${currentDesignTarget} xclbin golden_ref
    COMMENT "Running ${currentDesignTarget} with generated XCLBIN"
)

# add_aie_ci_test(${currentDesignTarget}
#                 RUN 
#                     "$<TARGET_FILE:${currentDesignTarget}> -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE --heads ${heads} --S_q ${S_q} --S_kv ${S_kv} -d ${d} -v 2 --warmup 1 --iters 1 --verbosity ${verbose}"
#                 CHECK
#                     "PASS!"
#                 METRICS 
#                     "Latency" [=[Latency \(us\): (?P<metric>\d+)]=]
#                     "Bandwidth" [=[Effective Bandwidth: (?P<metric>[\d\.e\+-]+) GB/s]=])
