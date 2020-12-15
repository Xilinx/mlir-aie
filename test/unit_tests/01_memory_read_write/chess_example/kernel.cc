// #include <stdlib.h>
// #include <string.h>
#include <stdio.h>
// #include <stdint.h>


//void func(int32_t *buf);

void func(int32_t *buf)
{
    int val=7;
    val = val+val;
    chess_report(val);
    buf[3] = val;
    val = 8;
    buf[5] = val;
}

int32_t buf[32];

int main()
{
    func(buf);
    //printf("test is %d\n",buf[8]);
}
