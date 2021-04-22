// #include <stdlib.h>
// #include <string.h>
#include <stdio.h>
// #include <stdint.h>

#define EAST_ID_BASE 48

//void func(int32_t *buf);

void func(int32_t *buf)
{
    acquire(EAST_ID_BASE+7,0);
    int tmp = ext_elem(srs(get_scd(),0),0);
    int val = tmp + tmp;
    val += tmp;
    val += tmp;
    val += tmp;
    buf[5] = val;
    release(EAST_ID_BASE+7,1);
}

int32_t buf[32];

int main()
{
    func(buf);
    //printf("test is %d\n",buf[8]);
}
