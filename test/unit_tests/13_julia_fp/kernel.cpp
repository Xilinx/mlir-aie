
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
#define VIDEO_LINE_WORDS 64

// liveout.
int error;

#define NUM_BITS 16
#define MAX_ITER 128

const int cols=64;
const int lines=64;

void do_line(unsigned *line_start_address,
	     float cr, float ci, float Im) {
	float MinRe = -2.0f, MaxRe = 2.0f;
	int index = 0;
	float Re = MinRe;
	int x;
	for (x=0; x < cols /*Re <= MaxRe*/; x++) {
		unsigned color;
		int done = 0;
		// struct pipe_state p = {};
	    float zr = Re, zi = Im;
		Re+= (MaxRe-MinRe)/cols;
		int n;
        color = 0xBEEF;
		for (n = 0; n < MAX_ITER && !done; n++) {
			float a, b;

			float x;
			a=zr*zr;
			b=zi*zi;
			if (((a+b) > 2.0f) && (done == 0)) {
				done = 1;
			}
			zi = zr*zi;
			zi = zi*2.0f + ci;
			zr = a-b + cr;
		}
		if(n > MAX_ITER) {
			color = 0xff;
		} else {
			color = n*(256*2/MAX_ITER);
		}
		line_start_address[x] = color;
	}
}

void julia(unsigned *framebuffer, float cr, float ci)
{
  // Fractal Julia code
  unsigned *line_start_address;
  float MinIm = -2.0f, MaxIm = 2.0f;
  line_start_address = framebuffer;
  float Im = MaxIm;
  for (int y=0; y < lines /*Im >= MinIm*/; Im -= (MaxIm-MinIm)/lines, y++) {
	  do_line(line_start_address, cr, ci, Im);
	  line_start_address += VIDEO_LINE_WORDS;
  }
}

void func(int32_t *a, int32_t *b)
{
   a[2] = 0xDEAD;
    julia((unsigned *)a, 0.3f, 0.3f);
    a[0] = 0xDEAD;
}
