#include "i32xi32.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IMAGE_SIZE                                                         \
  32768 // Test image are small (enough for 128x128 or 256x16)
#define MAX_KERNEL_SIZE 400 // Assume max kernel size is somethign like 20*20
#define VEC PARALLEL_FACTOR_32b // Vectorization factor

int32_t in_data[MAX_IMAGE_SIZE];
int32_t out_data[MAX_IMAGE_SIZE];
int32_t kernelCoeff[MAX_KERNEL_SIZE];

#define INPUT_FILE "data/kernelAndInImage_256x16_k3_gaussblur.txt"
#define REF_FILE "data/refImage_256x16_k3_gaussblur.txt"

int main() {
  int i = 0, j = 0, k = 0;

  // Open input and ref file
  FILE *fpin = NULL, *fpref = NULL;
  fpin = fopen(INPUT_FILE, "r");
  if (fpin == NULL) {
    printf("failure opening file %s for reading\n", INPUT_FILE);
    return -1;
  }
  fpref = fopen(REF_FILE, "r");
  if (fpref == NULL) {
    printf("failure opening file %s for reading\n", REF_FILE);
    return -1;
  }

  // Load kernel coefficients from input file
  int kernelSize;
  fscanf(fpin, "%d", &kernelSize);
  int index = 0;
  for (i = 0; i < kernelSize; i++) {
    for (j = 0; j < kernelSize; j++) {
      int val;
      fscanf(fpin, "%d", &val);
      kernelCoeff[index++] = val;
    }
  }
  kernelCoeff[index] = 0;

  // Print kernel coefficent values
  printf("kernel size: %d\n", kernelSize);
  for (i = 0; i < kernelSize * kernelSize + kernelSize; i++) {
    printf("%d ", kernelCoeff[i]);
  }
  printf("\n");

  // images.txt - File starts with the image size
  int imageWidth, imageHeight, imageChannels, borderPixels;
  fscanf(fpin, "%d", &imageWidth);
  fscanf(fpin, "%d", &imageHeight);
  fscanf(fpin, "%d", &imageChannels);
  fscanf(fpin, "%d", &borderPixels);
  int imageSize = imageWidth * imageHeight;

  int index_r_banked = 0;
  int index_w_seq = 0;
  int bank_index = 0;
  int pad = ((VEC - ((imageWidth + kernelSize - 1) % VEC)) % VEC) +
            VEC; // Padd image to nearest VEC+1
  int stride = imageWidth + kernelSize - 1 + pad;

  // Print image size
  printf("width: %d, height: %d, stride: %d\n", imageWidth, imageHeight,
         stride);

  // Read in image to local memory
  for (k = 0; k < imageHeight + kernelSize - 1; k++) {
    for (j = 0; j < imageWidth + kernelSize - 1; j++) {
      int val;
      fscanf(fpin, "%d", &val);
      in_data[index_r_banked++] = val;
    }
    for (j = 0; j < pad; j++) {
      in_data[index_r_banked++] = 0;
    }
  }

  // Compute filter2D
  // reference(in_data, kernelCoeff, out_data);
  conv2d(in_data, kernelCoeff, out_data);

  // Compare results with reference image
  // NOTE: There will be some rounding errors in results so we accept absolute
  // value differences < 5
  int refImageWidth, refImageHeight, refImageChannels, refBorderPixels;
  fscanf(fpref, "%d", &refImageWidth);
  fscanf(fpref, "%d", &refImageHeight);
  fscanf(fpref, "%d", &refImageChannels);
  fscanf(fpref, "%d", &refBorderPixels);
  int errors = 0;
  int max_error = 0;
  int absErrorDiff = 0;
  for (int i = 0; i < imageSize; i++) {
    int32_t ref_output;
    fscanf(fpref, "%d", &ref_output);
    if (out_data[i] != ref_output) {
      absErrorDiff = abs(out_data[i] - ref_output);
      if (absErrorDiff >= 5)
        printf("Delta found: Index %d is %d and should be %d\n", i, out_data[i],
               ref_output);
      if (absErrorDiff > max_error)
        printf("max found in index: %d\n", i);
      max_error = max_error < absErrorDiff ? absErrorDiff : max_error;
      errors++;
    }
  }
  if (errors == 0 || max_error < 5) {
    printf("PASSED, Max delta: %d, pixel intensity\n\n", max_error);
  } else {
    printf("FAIL. Number of deltas = %d, Max delta: %d, pixel intensity\n\n",
           errors, max_error);
  }

  fclose(fpin);
  fclose(fpref);

  return 0;
}
