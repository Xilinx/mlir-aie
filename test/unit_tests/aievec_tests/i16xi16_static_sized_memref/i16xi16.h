#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_16b 16 // Parallelization factor
#define SRS_SHIFT 10           // SRS shift

const int kernel_width = 3;
const int kernel_height = 3;

void conv2d(int16_t *restrict img_in, int16_t *restrict kernel_coeff,
            int16_t *restrict img_out);

void reference(int16_t *restrict img_in, int16_t *restrict kernel_coeff,
               int16_t *restrict img_out);

#endif
