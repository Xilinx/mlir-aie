# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Configuration variables
set(num_KV_heads 2)

set(heads 4)
set(S_q 1024)
set(S_kv 1024)
set(d 64)
set(B_q 64)
set(k 64)
set(B_kv 64)
set(dtype "bf16")

set(verbose 1)
set(trace_size 65536)
set(device_name "npu2")
set(design_name "mha")

set(ProjectName ${design_name})
set(currentDesignTarget ${design_name})

project(${ProjectName})

# Derived variables
set(target_suffix "${heads}_${S_q}_${S_kv}_${d}_${B_q}x${d}x${B_kv}")

set(mm_kernel "mm_${B_q}x${d}x${B_kv}_${dtype}_${dtype}")
set(softmax_kernel "softmax_${B_kv}_${dtype}_${dtype}")
set(mha_kernel "mha_${heads}_${S_q}_${S_kv}_${d}_${B_q}x${d}x${B_kv}_${dtype}")
set(passthrough_kernel "passthrough_${B_q}x${B_kv}_${dtype}")

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
message(STATUS "Design Configuration:")
message(STATUS "  Heads: ${heads}")
message(STATUS "  Number of KV Heads: ${num_KV_heads}")
message(STATUS "  Sequence Length Q: ${S_q}")
message(STATUS "  Sequence Length KV: ${S_kv}")
message(STATUS "  Embedding Dimension d: ${d}")
message(STATUS "  Tile dimensions: ${B_q}x${d}x${B_kv}")
message(STATUS "  Data type: ${dtype}")
message(STATUS "  Device: ${device_name}")
message(STATUS "  Design: ${design_name}")
message(STATUS "  Verbose mode: ${verbose}")

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
set(kernels_dir_aie2 "${CMAKE_SOURCE_DIR}/../../../../aie_kernels/aie2")
set(kernels_dir "${CMAKE_SOURCE_DIR}/../../../../aie_kernels/aie2p")
set(kernels_generic_dir "${CMAKE_SOURCE_DIR}/../../../../aie_kernels/generic")
set(runtime_lib_dir "${CMAKE_SOURCE_DIR}/../../../../runtime_lib")
set(test_lib_dir "${runtime_lib_dir}/test_lib")

find_program(aieopt_dir aie-opt)

if(aieopt_dir)
    # Get the parent directory of the binary
    get_filename_component(mlir_aie_bin_dir ${aieopt_dir} DIRECTORY)
    # Get the parent directory of the parent directory
    get_filename_component(mlir_aie_dir ${mlir_aie_bin_dir} DIRECTORY)
else()
    message(FATAL_ERROR "aie-opt not found in PATH")
endif()

# Build artifact paths
set(golden_ref_header "${CMAKE_BINARY_DIR}/golden_reference.h")
set(mlir_target "${CMAKE_BINARY_DIR}/aie_${target_suffix}.mlir")
set(trace_mlir_target "${CMAKE_BINARY_DIR}/aie_trace_${target_suffix}.mlir")
set(xclbin_target "${CMAKE_BINARY_DIR}/final_${target_suffix}.xclbin")
set(trace_xclbin_target "${CMAKE_BINARY_DIR}/trace_${target_suffix}.xclbin")
set(insts_target "${CMAKE_BINARY_DIR}/insts_${target_suffix}.txt")

set(mm_kernel_obj "${CMAKE_BINARY_DIR}/${mm_kernel}.o")
set(mm_kernel_obj_rowmaj "${CMAKE_BINARY_DIR}/${mm_kernel}_rowmaj.o")
set(softmax_kernel_obj "${CMAKE_BINARY_DIR}/${softmax_kernel}.o")
set(mha_kernel_obj "${CMAKE_BINARY_DIR}/${mha_kernel}.o")
set(passthrough_kernel_obj "${CMAKE_BINARY_DIR}/${passthrough_kernel}.o")

set(kernels_archive "${CMAKE_BINARY_DIR}/kernels.a")

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
    # "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"
)

# list(APPEND kernel_defines "-DB_COL_MAJ")
list(APPEND kernel_defines "-DBIT_WIDTH=16")

if(device_name STREQUAL "npu")
    set(kernel_cflags_base 
        "-O2" "-std=c++20" "--target=aie2-none-unknown-elf"
        ${warning_flags} "-DNDEBUG" "-I${mlir_aie_dir}/include"
    )
elseif(device_name STREQUAL "npu2")
    set(kernel_cflags_base 
        "-O2" "-std=c++20" "--target=aie2p-none-unknown-elf"
        ${warning_flags} "-DNDEBUG" "-I${mlir_aie_dir}/include"
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
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/generate_golden_reference.py
        --heads ${heads} --S_q ${S_q} --S_kv ${S_kv} -d ${d}
        --num_KV_heads ${num_KV_heads}
        --dtype ${dtype}
        --output ${golden_ref_header} ${verbose_string}
    DEPENDS ${CMAKE_SOURCE_DIR}/generate_golden_reference.py
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
        COMMAND ar rvs ${kernels_archive} ${mm_kernel_obj} ${mm_kernel_obj_rowmaj}_renamed ${softmax_kernel_obj} ${passthrough_kernel_obj} ${mha_kernel_obj}
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
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/${design_name}.py
        --heads ${heads}
        --S_q ${S_q} --S_kv ${S_kv} -d ${d}
        --B_q ${B_q} --B_kv ${B_kv}
        --num_KV_heads ${num_KV_heads}
        --trace_size 0 ${verbose_string} 
        --output_file_path ${mlir_target}
    DEPENDS ${CMAKE_SOURCE_DIR}/${design_name}.py
    COMMENT "Generating MLIR for ${target_suffix}"
)

# Trace MLIR generation
add_custom_command(
    OUTPUT ${trace_mlir_target}
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/${design_name}.py
        --heads ${heads}
        --S_q ${S_q} --S_kv ${S_kv} -d ${d}
        --B_q ${B_q} --B_kv ${B_kv}
        --num_KV_heads ${num_KV_heads}
        --trace_size ${trace_size} ${verbose_string} 
        --output_file_path ${mlir_target}
    DEPENDS ${CMAKE_SOURCE_DIR}/${design_name}.py
    COMMENT "Generating trace MLIR for ${target_suffix}"
)

# XCLBIN generation
add_custom_command(
    OUTPUT ${xclbin_target} ${insts_target}
    COMMAND ${aiecc_executable} ${aiecc_flags}
        --xclbin-name=final_${target_suffix}.xclbin
        --npu-insts-name=insts_${target_suffix}.txt
        ${mlir_target}
    DEPENDS ${mlir_target} ${kernels_archive} #${mm_kernel_obj}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating XCLBIN and instructions for ${target_suffix}"
)

# Trace XCLBIN generation
add_custom_command(
    OUTPUT ${trace_xclbin_target}
    COMMAND ${aiecc_executable} ${aiecc_flags}
        --xclbin-name=trace_${target_suffix}.xclbin
        --npu-insts-name=insts_${target_suffix}.txt
        ${trace_mlir_target}
    DEPENDS ${trace_mlir_target} ${mm_kernel_obj}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating trace XCLBIN for ${target_suffix}"
)

# Host executable
add_executable(${currentDesignTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../runtime_lib/test_lib/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

target_compile_definitions(${currentDesignTarget} PUBLIC 
    DISABLE_ABI_CHECK=1
)

target_include_directories (${currentDesignTarget} PUBLIC
    ${XRT_INC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../runtime_lib
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
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

# Ensure golden reference is generated before compiling
add_dependencies(${currentDesignTarget} golden_ref)

# Custom targets
add_custom_target(golden_ref DEPENDS ${golden_ref_header})
add_custom_target(xclbin DEPENDS ${xclbin_target} ${insts_target})
add_custom_target(trace_xclbin DEPENDS ${trace_xclbin_target})

# Run target
add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env XRT_HACK_UNSECURE_LOADING_XCLBIN=1
        $<TARGET_FILE:${currentDesignTarget}>
        -x ${xclbin_target}
        -i ${insts_target}
        -k MLIR_AIE
        --heads ${heads}
        --S_q ${S_q} --S_kv ${S_kv} -d ${d}
        -v 2 --warmup 1 --iters 1
        --verbosity ${verbose}
    DEPENDS ${currentDesignTarget} xclbin
    COMMENT "Running ${currentDesignTarget} test"
)

# Trace target
add_custom_target(trace
    COMMAND ${CMAKE_COMMAND} -E env XRT_HACK_UNSECURE_LOADING_XCLBIN=1
        $<TARGET_FILE:${currentDesignTarget}>
        -x ${trace_xclbin_target}
        -i ${insts_target}
        -k MLIR_AIE
        --heads ${heads}
        --S_q ${S_q} --S_kv ${S_kv} -d ${d}
        -v 2 --warmup 1 --iters 1
        -t ${trace_size}
        --verbosity ${verbose}
    DEPENDS ${currentDesignTarget} trace_xclbin
    COMMENT "Running ${currentDesignTarget} test with tracing"
)

# Parse trace target
add_custom_target(parse_trace
    COMMAND ${python_executable} ${CMAKE_SOURCE_DIR}/../../utils/parse_trace.py
        --input trace.txt
        --mlir ${trace_mlir_target}
        --output trace_mm.json
    COMMENT "Parsing trace output"
)

# Clean targets
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build artifacts"
)
