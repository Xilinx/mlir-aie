#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024  Advanced Micro Devices, Inc.


set(LLVM_TARGET_DEFINITIONS CIRToAIEPasses.td)
mlir_tablegen(CIRToAIEPasses.h.inc -gen-pass-decls -name CIRToAIE)
add_public_tablegen_target(MLIRCIRToAIEPassesIncGen)

add_mlir_doc(CIRToAIEPasses CIRToAIEPasses ./ -gen-pass-doc)
