#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AIEVecAnalysis)
add_public_tablegen_target(MLIRAIEVecAnalysisPassIncGen)

add_mlir_doc(Passes AIEVecAnalysisPasses ./ -gen-pass-doc)
