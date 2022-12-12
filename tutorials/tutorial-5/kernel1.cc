#define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel1() {
    v16int16 v16 = null_v16int16();
    v16 = upd_elem(v16,0,14);
    put_mcd(v16);
}

} // extern "C"