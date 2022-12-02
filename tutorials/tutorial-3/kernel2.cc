#define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel2(int32_t *restrict buf) {
    buf[5] = buf[3] + 100;
}

} // extern "C"