# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_library(AIECAPI
Dialects.cpp
Registration.cpp

DEPENDS

ENABLE_AGGREGATION
LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
AIE
ADF
MLIRAIEVec
#AIEInitAll
MLIRIR
MLIRSupport
)