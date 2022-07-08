#include "adf.h"

void add(input_window_int32* in, output_window_int32* out, int param)
{
    for (int i=0; i<32; i++)
    {
        int val = window_readincr(in);
        val = val + param;
        window_writeincr(out, val);
    }
}

/*void add(input_window_int32* in, int (&out)[8], int param)
{
    for (int i=0; i<8; i++)
    {
        int val = window_readincr(in);
        val = val + param;
        out[i] = val;
    }
}*/