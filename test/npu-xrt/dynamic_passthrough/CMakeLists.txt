#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# Test for runtime transaction generation using standalone library

add_executable(test_runtime_generation test_runtime_generation.cpp)

# Include path for standalone npu_instructions library
target_include_directories(test_runtime_generation PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime_lib
)

# This is a standalone test - no MLIR dependencies needed
target_compile_features(test_runtime_generation PRIVATE cxx_std_17)

# Add test
add_test(NAME RuntimeTxnGeneration
         COMMAND test_runtime_generation
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
