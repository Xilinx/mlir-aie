#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2022 Xilinx Inc.


set(LLVM_TARGET_DEFINITIONS AIEXPasses.td)
mlir_tablegen(AIEXPasses.h.inc -gen-pass-decls -name AIEX)
add_public_tablegen_target(MLIRAIEXPassIncGen)

add_mlir_doc(AIEXPasses AIEXPasses ./ -gen-pass-doc)
