#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(int16_t *__restrict mat_a_data, size_t m1,
            int16_t *__restrict mat_b_data, size_t m2,
            int16_t *__restrict mat_c_data, size_t m3);
