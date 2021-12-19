//===- kernel.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

/**
This core computes a sequence of julia set images.  This set is computed by
iterating a complex function, defined by: Z(k+1) = Z(k) + c. Each
pixel in the image corresponds to the result of applying the iteration
to a corresponding complex number Z(0).  If the iteration does not
grow without bound, then the pixel is colored white, otherwise the
pixel is shaded according to how fast the iteration grows.

Because of recurrence defined by the complex iteration, a simple
implementation results in poor computational efficiency on the data
path, which has a latency of about 7 cycles when implemented using
DSP48s in FPGA.  This implementation time multiplexes the computation
of 8 pixels at a time on a single datapath in order to increase
throughput.  When the iteration has completed for all 8 pixels, they
are written out to a frame buffer in external memory.  In a system,
this frame buffer can be read by a separate core (such as \ref
npi64_frame_buffer_output) to display the images.
 */

#include <stdint.h>

/** Video lines are separated by this many 4-byte words in a frame buffer. */
#define VIDEO_LINE_WORDS 32

// liveout.
int error;

#define NUM_BITS 16
#define MAX_ITER 255


extern float *debuf;

__attribute__((noinline)) void do_line(int32_t *line_start_address, float MinRe, float StepRe, float Im, int cols) {
	int index = 0;
	float Re = MinRe;
	int x;
	for (x=0; x < cols /*Re <= MaxRe*/; x++) {
		int32_t color;
		int done = 0;
		// struct pipe_state p = {};
	    float zr = Re, zi = Im;
		Re += StepRe;
		int n = 0;
		float cr = zr; float ci = zi; // For Mandelbrot
        color = 0xBEEF;
//		debuf[x] = Re;
		for (n = 0; n < MAX_ITER && !done; n++) {
			float a, b;

			a=zr*zr;
			b=zi*zi;
			if (((a+b) > 2.0f) && (done == 0)) {
				done = 1;
			}
			zi = zr*zi;
			zi = zi*2.0f + ci;
			zr = a-b + cr;
		//	_b[x] = Im;
		}
		if(n >= MAX_ITER) {
			color = 0xff;
		} else {
			color = n*(255/MAX_ITER);
		}
		line_start_address[x] = n;
	}
}

// __attribute__((noinline)) static void julia(int32_t *framebuffer, float MinRe, float MinIm, float StepRe, float StepIm, float cr, float ci)
// {
//   // Fractal Julia code
//   int32_t *line_start_address;
//   line_start_address = framebuffer;
//   float Im = MinIm;
//   for (int y = 0; y < lines; Im += StepIm, y++) {
// //	  do_line(line_start_address, MinRe, StepRe, cr, ci, Im);
// 	  line_start_address += VIDEO_LINE_WORDS;
// 	  framebuffer[0] = y;
//   }
//   framebuffer[0] = lines;
// }

// void func(int32_t *a, float MinRe, float MaxRe, float MinIm, float MaxIm)
// {
//   float StepRe = (MaxRe-MinRe)/cols;
//   float StepIm = (MaxIm-MinIm)/lines;
//   julia(a, MinRe, MinIm, StepRe, StepIm, 0.3f, 0.3f);
// }
