#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.


set(LLVM_TARGET_DEFINITIONS AIEPasses.td)
mlir_tablegen(AIEPasses.h.inc -gen-pass-decls -name AIE)
add_public_tablegen_target(MLIRAIEPassIncGen)

add_mlir_doc(AIEPasses AIEPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS AIENormalizeAddressSpaces.td)
mlir_tablegen(AIENormalizeAddressSpaces.inc -gen-rewriters)
add_public_tablegen_target(MLIRAIENormalizeAddressSpacesIncGen)
