// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_LOG_Q_H
#define AIE_KERNELS_COMMON_LOG_Q_H

// log(m) for m in [sqrt(1/2), sqrt(2)) is f q(f) with f = m - 1, and q is
// log1p(f) / f interpolated at the 10 Chebyshev nodes of that range (numpy
// polyfit, degree 9): 2.5e-9 from log1p(f) with these float32 coefficients.
static const float log_q[10] = {1.0f,
                                -0.49999988079071045f,
                                0.33333346247673035f,
                                -0.2500157952308655f,
                                0.20000939071178436f,
                                -0.1660836935043335f,
                                0.14199650287628174f,
                                -0.13266031444072723f,
                                0.12806610763072968f,
                                -0.07451186329126358f};

#endif
