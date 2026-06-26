/*
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#pragma once

#include "adf.h"

void add(input_window_int32* in, output_window_int32* out, int param);
//void add(input_window_int32* in, int (&out)[8], int param);