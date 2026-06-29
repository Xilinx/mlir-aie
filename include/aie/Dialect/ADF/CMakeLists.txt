#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect(ADF ADF)
add_mlir_doc(ADF ADFDialect ./ -gen-dialect-doc -dialect=ADF)
set(LLVM_TARGET_DEFINITIONS ADF.td)
