# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# Common CMake configuration for programming guide examples
# This file provides common setup for test_utils library linking

# Get MLIR_AIE_DIR from Python
execute_process(
    COMMAND python3 -c "from aie.utils.config import root_path; print(root_path())"
    OUTPUT_VARIABLE MLIR_AIE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set test_utils library paths
set(TEST_UTILS_LIB_DIR "${MLIR_AIE_DIR}/runtime_lib/x86_64/test_lib/lib")
set(TEST_UTILS_INC_DIR "${MLIR_AIE_DIR}/runtime_lib/x86_64/test_lib/include")
set(TEST_UTILS_RUNTIME_LIB_DIR "${MLIR_AIE_DIR}/runtime_lib")

# Function to add test_utils library to a target
function(target_link_test_utils target_name)
    target_include_directories(${target_name} PUBLIC
        ${TEST_UTILS_RUNTIME_LIB_DIR}
        ${TEST_UTILS_INC_DIR}
    )
    
    target_link_directories(${target_name} PUBLIC
        ${TEST_UTILS_LIB_DIR}
    )
    
    target_link_libraries(${target_name} PUBLIC
        test_utils
    )
endfunction()
