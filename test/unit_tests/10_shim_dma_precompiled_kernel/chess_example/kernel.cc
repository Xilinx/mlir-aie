#include <stdio.h>

int32_t iter;

int32_t a_ping[256];
int32_t a_pong[256];
int32_t b_ping[256];
int32_t b_pong[256];

#define LOCK_OFFSET 48
#define A_PING (LOCK_OFFSET+3)
#define A_PONG (LOCK_OFFSET+4)
#define B_PING (LOCK_OFFSET+5)
#define B_PONG (LOCK_OFFSET+6)
#define LOCK_READ  1
#define LOCK_WRITE 0


inline void func(int32_t *a, int32_t *b)
{
    int val=a[3];
    int val2=val+val;
    val2 += val;
    val2 += val;
    val2 += val;
    b[5] = val2;
}

void func_wrap()
{
    int bounds = iter;

    // NOTE: odd iterations need locks reset externally when core is run again
    while(bounds > 0) { // TODO: need to change this to start count at 0 so we do ping first
        if((bounds & 0x1) == 0) {
            acquire(A_PING,LOCK_READ); 
            acquire(B_PING,LOCK_WRITE); 
            func(&a_ping[0], &b_ping[0]);
            release(A_PING,LOCK_WRITE); 
            release(B_PING,LOCK_READ); 
        } else {
            acquire(A_PONG,LOCK_READ); 
            acquire(B_PONG,LOCK_WRITE); 
            func(&a_pong[0], &b_pong[0]);
            release(A_PONG,LOCK_WRITE); 
            release(B_PONG,LOCK_READ); 
        }
      bounds--;
    }
}

int main()
{
    func_wrap();
    //printf("test is %d\n",buf[8]);
}
