#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.

add_mlir_dialect(ADF ADF)
add_mlir_doc(ADF ADFDialect ./ -gen-dialect-doc -dialect=ADF)
set(LLVM_TARGET_DEFINITIONS ADF.td)
