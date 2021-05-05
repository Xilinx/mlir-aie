// #include <stdlib.h>
// #include <string.h>
#include <stdio.h>
// #include <stdint.h>


//void func(int32_t *buf);

extern int32_t a[256];
extern int32_t b[256];

#define LOCK_OFFSET 48

void func()
{
    acquire(LOCK_OFFSET+3,1);
    acquire(LOCK_OFFSET+5,0);
    int val=a[3];
    int val2=val+val;
    val2 += val;
    val2 += val;
    val2 += val;
    b[5] = val2;
    release(LOCK_OFFSET+3,0);
    release(LOCK_OFFSET+5,1);
}

int main()
{
    func();
    //printf("test is %d\n",buf[8]);
}
