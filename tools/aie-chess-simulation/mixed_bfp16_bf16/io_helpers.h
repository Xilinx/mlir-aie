/*  (c) Copyright 2014 - 2019 Xilinx, Inc. All rights reserved.

    This file contains confidential and proprietary information
    of Xilinx, Inc. and is protected under U.S. and
    international copyright and other intellectual property
    laws.

    DISCLAIMER
    This disclaimer is not a license and does not grant any
    rights to the materials distributed herewith. Except as
    otherwise provided in a valid license issued to you by
    Xilinx, and to the maximum extent permitted by applicable
    law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
    WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
    AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
    BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
    INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
    (2) Xilinx shall not be liable (whether in contract or tort,
    including negligence, or under any other theory of
    liability) for any loss or damage of any kind or nature
    related to, arising under or in connection with these
    materials, including for any direct, or any indirect,
    special, incidental, or consequential loss or damage
    (including loss of data, profits, goodwill, or any type of
    loss or damage suffered as a result of any action brought
    by a third party) even if such damage or loss was
    reasonably foreseeable or Xilinx had been advised of the
    possibility of the same.

    CRITICAL APPLICATIONS
    Xilinx products are not designed or intended to be fail-
    safe, or for use in any application requiring fail-safe
    performance, such as life-support or safety devices or
    systems, Class III medical devices, nuclear facilities,
    applications related to the deployment of airbags, or any
    other applications that could lead to death, personal
    injury, or severe property or environmental damage
    (individually and collectively, "Critical
    Applications"). Customer assumes the sole risk and
    liability of any use of Xilinx products in Critical
    Applications, subject only to applicable laws and
    regulations governing limitations on product liability.

    THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
    PART OF THIS FILE AT ALL TIMES.                       */

#pragma once

#ifndef __AIE_API_TESTS_IO_HELPERS_HPP__
#define __AIE_API_TESTS_IO_HELPERS_HPP__

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "aie_api/aie.hpp"

[[maybe_unused]] static FILE *open_file(const char* filename, const char *mode)
{
    FILE *fp = fopen(filename,mode);

    if (fp == NULL) {
        fprintf(stderr, "ERROR: Cannot open file '%s'.\n",filename);
        exit(1);
    }

    return fp;
}

[[maybe_unused]] static void write_file(const int8 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%d\n", output[i]);

    fclose(fp);
}

[[maybe_unused]] static void write_file(const uint8 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%u\n", (unsigned)output[i]);

    fclose(fp);
}

[[maybe_unused]] static void write_file(const int16 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%d\n", output[i]);

    fclose(fp);
}

[[maybe_unused]] static void write_file(const uint16 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%u\n", output[i]);

    fclose(fp);
}

[[maybe_unused]] static void write_file(const int32 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%d\n", output[i]);

    fclose(fp);
}

[[maybe_unused]] static void write_file(const uint32 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%u\n", output[i]);

    fclose(fp);
}

[[maybe_unused]] static void write_file(float *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%f\n", output[i]);

    fclose(fp);
}

#if __AIE_ARCH__ >= 20
[[maybe_unused]] static void write_file(bfloat16 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");

    for (int i = 0; i < num; i++)
        fprintf(fp, "%f\n", (float)(output[i]));

    fclose(fp);
}
#endif

[[maybe_unused]] static void write_file(const float *output, unsigned num, bool cmplx, const char* filename)
{
    FILE *fp = open_file(filename,"w+");
    if (cmplx) {
        for (int i = 0; i < num/2; i++)
            fprintf(fp, "%9.6g %9.6g\n", output[2*i], output[2*i+1]);
    }
    else {
        for (int i = 0; i < num; i++)
            fprintf(fp, "%f\n", output[i]);
    }

    fclose(fp);
}

#if __AIE_ARCH__ == 10 || __AIE_API_COMPLEX_FP32_EMULATION__
[[maybe_unused]] static void write_file(const cfloat *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename,"w+");
    const float *tmp = (const float*)output; 

    for (int i = 0; i < num; i++)
        fprintf(fp, "%9.6g %9.6g\n", tmp[2*i], tmp[2*i+1]);

    fclose(fp);
}
#endif

[[maybe_unused]] static void read_file(int8 *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        int re;
        int ret = fscanf(fp, "%d", &re);
        if (ret != 1) fprintf(stderr, "failed: %d\n", i);
        assert(ret == 1);

        *dest++ = re;
    }

    fclose(fp);
}

[[maybe_unused]] static void read_file(uint8 *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        unsigned re;
        int ret = fscanf(fp, "%u", &re);
        if (ret != 1) fprintf(stderr, "failed: %d\n", i);
        assert(ret == 1);

        *dest++ = re;
    }

    fclose(fp);
}

[[maybe_unused]] static void read_file(int16 *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        int re;
        int ret = fscanf(fp, "%d", &re);
        assert(ret == 1);

        *dest++ = re;
    }

    fclose(fp);
}

[[maybe_unused]] static void read_file(uint16 *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        unsigned re;
        int ret = fscanf(fp, "%u", &re);
        assert(ret == 1);

        *dest++ = re;
    }

    fclose(fp);
}

[[maybe_unused]] static void read_file(int32 *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        int re;
        int ret = fscanf(fp, "%d", &re);
        assert(ret == 1);

        *dest++ = re;
    }

    fclose(fp);
}

[[maybe_unused]] static void read_file(uint32 *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        unsigned re;
        int ret = fscanf(fp, "%u", &re);
        assert(ret == 1);

        *dest++ = re;
    }

    fclose(fp);
}

typedef int (*stream_32_in_t)();
typedef void (*stream_32_out_t)(int);

//read 32bit stream
[[maybe_unused]] static void read_stream(int16 *dest, unsigned num, bool cplx, stream_32_in_t stream_in)
{
    int32 tmp;
    if (cplx) {
        for (int i=0; i<num; i++) {
            tmp=(*stream_in)();
            *dest++=(short)(tmp&0xffff);
            *dest++=(short)((tmp>>16)&0xffff);
        }
    }
    else {
        for (int i=0; i<num; i++) {
            tmp=(*stream_in)();
            *dest++=(short)(tmp&0xffff);
        }
    }
}

[[maybe_unused]] static void write_file(const cint16 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "w+");

    for (int i = 0; i < num; ++i)
        fprintf(fp,"%d\t%d\n",(short)output[i].real,(short)output[i].imag);

    fclose(fp);
}

[[maybe_unused]] static void write_file(const cint32 *output, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "w+");

    for (int i = 0; i < num; ++i)
        fprintf(fp,"%d\t%d\n",output[i].real,output[i].imag);

    fclose(fp);
}

[[maybe_unused]] static void read_file(cint16 *dest, unsigned num, const char* filename)
{
    read_file((int16 *)dest,num*2,filename);
}

[[maybe_unused]] static void stream_output_data(cint16 *output, int size, stream_32_out_t stream_out)
{
    for (int i=0; i<size; i++)
        (*stream_out)(*(int*)(&output[i]));
}

[[maybe_unused]] static void read_file(cint32 *dest, unsigned num, const char* filename)
{
    read_file((int32 *)dest,num*2,filename);
}

[[maybe_unused]] static void read_file(float *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        float re;
        int ret = fscanf(fp, "%f", &re);
        assert(ret == 1);

        *dest++ = re;
    }

    fclose(fp);
}

#if __AIE_ARCH__ >= 20
[[maybe_unused]] static void read_file(bfloat16 *dest, unsigned num, const char* filename)
{
    FILE *fp = open_file(filename, "r");

    for (int i = 0; i < num; ++i) {
        float re;
        int ret = fscanf(fp, "%f", &re);
        assert(ret == 1);

        *dest++ = (bfloat16)re;
    }

    fclose(fp);
}
#endif

[[maybe_unused]] static void read_file(cfloat *dest, unsigned num, const char* filename)
{
    return read_file((float *)dest, num * 2, filename);
}

#endif // __AIE_API_TESTS_IO_HELPERS_HPP__
