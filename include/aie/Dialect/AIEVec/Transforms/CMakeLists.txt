#
# Copyright (C) 2022 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AIEVec)
add_public_tablegen_target(MLIRAIEVecPassIncGen)

add_mlir_doc(Passes AIEVecPasses ./ -gen-pass-doc)
