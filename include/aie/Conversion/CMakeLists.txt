# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header -prefix Conversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl -prefix Conversion)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIEConversionPassIncGen)

add_mlir_doc(Passes MLIRAIEConversionPasses ./ -gen-pass-doc)
