#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void gemm_64x32x64_bf16_packed_4x8x4(bfloat16 *restrict mat_a_data,
                                     bfloat16 *restrict mat_b_data,
                                     float *restrict mat_c_data);
