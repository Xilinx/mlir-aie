#define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel1(int32_t *restrict buf) {
    buf[3] = 14;
}

} // extern "C"