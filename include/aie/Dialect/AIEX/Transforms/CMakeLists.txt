#
# Copyright (C) 2022 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


set(LLVM_TARGET_DEFINITIONS AIEXPasses.td)
mlir_tablegen(AIEXPasses.h.inc -gen-pass-decls -name AIEX)
add_public_tablegen_target(MLIRAIEXPassIncGen)

add_mlir_doc(AIEXPasses AIEXPasses ./ -gen-pass-doc)
