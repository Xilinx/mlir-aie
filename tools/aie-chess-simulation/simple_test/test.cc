#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

int __inline__  kernel(int a)
{
    return a*2;
}



int main()
{
    printf("test start ...\n");
    int a = 1;
    int res = kernel(a);

    printf("test done!\n");
    return 0;
}
