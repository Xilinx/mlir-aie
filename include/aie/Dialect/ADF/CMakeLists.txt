#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

add_mlir_dialect(ADF ADF)
add_mlir_doc(ADF ADFDialect ./ -gen-dialect-doc -dialect=ADF)
set(LLVM_TARGET_DEFINITIONS ADF.td)
