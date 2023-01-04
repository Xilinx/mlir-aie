#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_16b 16 // Parallelization factor
#define SRS_SHIFT 10           // SRS shift

const int kernel_width = 1;
const int kernel_height = 3;

void conv2d(int16_t *__restrict img_in, size_t m1,
            int16_t *__restrict kernel_coeff, size_t m2,
            int16_t *__restrict img_out, size_t m3);

#endif
