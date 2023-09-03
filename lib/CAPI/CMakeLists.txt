# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_public_c_api_library(AIECAPI
Dialects.cpp
Registration.cpp

LINK_LIBS PUBLIC
AIE
ADF
MLIRAIEVec
)