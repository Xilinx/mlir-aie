#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.

add_mlir_dialect(AIEX aiex)
add_mlir_doc(AIEX AIEXDialect ./ -gen-dialect-doc -dialect=aiex)

# AIEX.td pulls in AIEAttrs.td which includes generated event enums
add_dependencies(MLIRAIEXIncGen GenerateAIEEventsTD)

# Add AIEX enums
set(LLVM_TARGET_DEFINITIONS AIEXAttrs.td)
mlir_tablegen(AIEXEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIEXEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIEXEnumsIncGen)
add_dependencies(MLIRAIEXIncGen MLIRAIEXEnumsIncGen)
add_dependencies(aie-headers MLIRAIEXEnumsIncGen)
