#include "kernel.h"

    
void pass(int *a, int *b)
{
    for (int i = 0; i < 64; i ++){
        b[i] = a[i];
    }
}