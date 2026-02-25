#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_aie_runtime_libs(AIE2)

if(AIETools_AIE2_FOUND OR Peano_AIE2_FOUND)
    compile_aie_runtime_libs(AIE2)
endif()
