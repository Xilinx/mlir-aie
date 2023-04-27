#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(int16_t *__restrict mat_a_data, int16_t *__restrict mat_b_data,
            int16_t *__restrict mat_c_data);
