#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

# Test for 8x8x8 volume with 8 channels
add_test(NAME conv3d_8x8x8
         COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} run_py
                 depth=8 height=8 width=8 in_channels=8 out_channels=8
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
