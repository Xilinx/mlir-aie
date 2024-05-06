# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

add_executable(example_noop_test example_noop_test.cpp)

target_link_libraries(example_noop_test PRIVATE xrt_coreutil dl)
target_include_directories(example_noop_test PRIVATE ${XRT_INCLUDE_DIR})

target_link_libraries(example_noop_test PRIVATE xaienginecdo_static)
add_dependencies(example_noop_test xaienginecdo_static xaienginecdo_static-headers)