#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_FLOAT 8 // Parallelization factor
#define SRS_SHIFT 10            // SRS shift

const int kernel_width = 3;
const int kernel_height = 3;

void reference(float *restrict img_in, float *restrict kernel_coeff,
               float *restrict img_out);

void conv2d(float *restrict img_in, float *restrict kernel_coeff,
            float *restrict img_out);

#endif
