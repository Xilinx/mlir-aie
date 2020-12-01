#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

int test(int a)
{
    return a*2;
}

int main()
{
    int ret = test(10);
    printf("test is %d\n",ret);
}
