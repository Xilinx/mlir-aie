#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_32b 8 // Parallelization factor
#define SRS_SHIFT 10          // SRS shift

const int kernel_width = 3;
const int kernel_height = 3;

void reference(int32_t *restrict img_in, int32_t *restrict kernel_coeff,
               int32_t *restrict img_out);

void conv2d(int32_t *restrict img_in, int32_t *restrict kernel_coeff,
            int32_t *restrict img_out);

#endif
