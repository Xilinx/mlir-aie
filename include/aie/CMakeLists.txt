#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.

configure_file(version.h.in version.h)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
