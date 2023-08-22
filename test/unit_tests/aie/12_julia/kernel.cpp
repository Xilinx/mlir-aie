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

//#include <stdio.h>
#include <stdint.h>

/** Video lines are separated by this many 4-byte words in a frame buffer. */
#define VIDEO_LINE_WORDS 64

// liveout.
int error;
//unsigned int count;
// struct pipe_state {
// 	int zr, zi;
// 	unsigned8_t color;
//    bool done;
// };

#define NUM_BITS 16
#define MAX_ITER 128

// void init_state(struct pipe_state &s, int Re, int Im) {
// 	s.zr = Re;
// 	s.zi = Im;
// 	s.color = 0xff;
// 	s.done = 0;
// }

// unsigned int iterate_state(int n, struct pipe_state &s, int cr, int ci) {
// 	int a, b;

// 	int64_t x;
// 	a=((int64_t)s.zr*(int64_t)s.zr)>>NUM_BITS;
// 	b=((int64_t)s.zi*(int64_t)s.zi)>>NUM_BITS;
// 	if (((a+b) > 0x40000) && (s.done == 0)) {
// 		if(n > MAX_ITER/2) {
// 			s.color = 0xff;
// 		} else {
// 			s.color = n*(256*2/MAX_ITER);
// 		}
// 		s.done = 1;
// 	}
// 	s.zi = ((int64_t)s.zr*(int64_t)s.zi)>>NUM_BITS;
// 	s.zi = s.zi*2 + ci;
// 	s.zr = a-b+cr;
// 	return s.done;
// }
// inline unsigned int iterate_state(int n, int &zr, int &zi, int cr, int ci) {
// 	int a, b;

// 	int64_t x;
//     int done = 0;
// 	a=((int64_t)zr*(int64_t)zr)>>NUM_BITS;
// 	b=((int64_t)zi*(int64_t)zi)>>NUM_BITS;
// 	if (((a+b) > 0x40000) && (done == 0)) {
// 		done = 1;
// 	}
// 	zi = ((int64_t)zr*(int64_t)zi)>>NUM_BITS;
// 	zi = zi*2 + ci;
// 	zr = a-b + cr;
// 	return done;
// }

// void write_array(int flag, unsigned a, unsigned d[8]) {
// 	unsigned d1, d2, d3, d4;
// 	if(flag == 0) {
// 		write_8_words(a, d[0], d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
// 	} else {
// 		// This should never happen.
// 		struct npi32_read_data d = read_data();
// 	}
// }

// #define BLOCK_SIZE 1
// void do_block(int &Re, int Im, int cr, int ci, unsigned color[BLOCK_SIZE]) {
// 	for(int8_t k = 0; k < BLOCK_SIZE; k++) {
// 		struct pipe_state p = {};

// 		int done = 0;

// 		for (int9_t n = 0; n < MAX_ITER && !done; n++) {

// 			if(n == 0) {
// 				init_state(p, Re, Im);
// 				Re+=0xCC;
// 			}
// 			done = iterate_state(n, p, cr, ci);

// 			struct pixel_data pixel;
// 			pixel.r = p.color;
// 			pixel.g = p.color;
// 			pixel.b = p.color;

// 			color[k] = pack_pixel(pixel);
// 		}
// 	}
// }

const int cols=64;
const int lines=64;

void do_line(unsigned *line_start_address,
	     int cr, int ci, int Im) {
	int MinRe = 0xFFFE0000, MaxRe = 0x00020000;
	int index = 0;
	int Re = MinRe;
	int x;
	for (x=0; x < cols /*Re <= MaxRe*/; x++) {
		unsigned color;
		int done = 0;
		// struct pipe_state p = {};
		int zr = Re, zi = Im;
		Re+= (MaxRe-MinRe)/cols;
		int n;
        color = 0xBEEF;
        for (n = 0; n < MAX_ITER && !done; n++) {
          int a, b;

          int64_t x;
          a = ((int64_t)zr * (int64_t)zr) >> NUM_BITS;
          b = ((int64_t)zi * (int64_t)zi) >> NUM_BITS;
          if (((a + b) > 0x40000) && (done == 0)) {
            done = 1;
          }
          zi = ((int64_t)zr * (int64_t)zi) >> NUM_BITS;
          zi = zi * 2 + ci;
          zr = a - b + cr;
        }
        if (n > MAX_ITER) {
          color = 0xff;
        } else {
          color = n * (256 * 2 / MAX_ITER);
        }
        line_start_address[x] = color;
        }
}

void julia(unsigned *framebuffer, int cr, int ci)
{
  // Fractal Julia code
  unsigned *line_start_address;
  int MinIm = 0xFFFE0000, MaxIm = 0x00020000;
  line_start_address = framebuffer;
  int Im = MaxIm;
  for (int y = 0; y < lines /*Im >= MinIm*/;
       Im -= (MaxIm - MinIm) / lines, y++) {
    do_line(line_start_address, cr, ci, Im);
    line_start_address += VIDEO_LINE_WORDS;
  }
}

void func(int32_t *a, int32_t *b) { julia((unsigned *)b, a[0], a[1]); }
