#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(int32_t *__restrict mat_a_data, int32_t *__restrict mat_b_data,
            int32_t *__restrict mat_c_data);
