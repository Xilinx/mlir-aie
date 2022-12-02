#ifndef _MY_KERNEL_H
#define _MY_KERNEL_H

//extern "C" {
void extern_kernel(int32_t *restrict A, int32_t *restrict B,
                   int32_t *restrict acc, int32_t *restrict C);
//}

#endif

