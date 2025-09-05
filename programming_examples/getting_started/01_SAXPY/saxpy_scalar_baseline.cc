#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

extern "C" {
void saxpy(bfloat16 *x, bfloat16 *y, bfloat16 *z) {
  event0();
  bfloat16 a = 3f;
  for (int i = 0; i < 4096; ++i) {
    z[i] = a * x[i] + y[i];
  }
  event1();
}
}
