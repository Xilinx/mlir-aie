//  (c) Copyright 2019-2022 Xilinx, Inc. All rights reserved.
//  (c) Copyright 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
//
//  This file contains confidential and proprietary information
//  of Xilinx, Inc. and is protected under U.S. and
//  international copyright and other intellectual property
//  laws.
//
//  DISCLAIMER
//  This disclaimer is not a license and does not grant any
//  rights to the materials distributed herewith. Except as
//  otherwise provided in a valid license issued to you by
//  Xilinx, and to the maximum extent permitted by applicable
//  law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
//  WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
//  AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
//  BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
//  INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
//  (2) Xilinx shall not be liable (whether in contract or tort,
//  including negligence, or under any other theory of
//  liability) for any loss or damage of any kind or nature
//  related to, arising under or in connection with these
//  materials, including for any direct, or any indirect,
//  special, incidental, or consequential loss or damage
//  (including loss of data, profits, goodwill, or any type of
//  loss or damage suffered as a result of any action brought
//  by a third party) even if such damage or loss was
//  reasonably foreseeable or Xilinx had been advised of the
//  possibility of the same.
//
//  CRITICAL APPLICATIONS
//  Xilinx products are not designed or intended to be fail-
//  safe, or for use in any application requiring fail-safe
//  performance, such as life-support or safety devices or
//  systems, Class III medical devices, nuclear facilities,
//  applications related to the deployment of airbags, or any
//  other applications that could lead to death, personal
//  injury, or severe property or environmental damage
//  (individually and collectively, "Critical
//  Applications"). Customer assumes the sole risk and
//  liability of any use of Xilinx products in Critical
//  Applications, subject only to applicable laws and
//  regulations governing limitations on product liability.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//-------------------------------------
// Group 0a, Gather
//-------------------------------------
void magika_group0a(short xin[2048], float lkup[257 * 64],
                    float yout[256 * 512]) {
  // temp buffer for lookup
  float temp[2048][64];

  // loop for all inputs
  for (int i = 0; i < 2048; i++) {
    short idx = xin[i];
    // look up the dictionary for vector of 64 elements
    for (int j = 0; j < 64; j++)
      temp[i][j] = lkup[idx * 64 + j];
  }

  // shuffle the temp buffer for output
  // divide 2048 inputs into 4 groups of 512
  for (int i = 0; i < 512; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 64; k++)
        yout[(j * 64 + k) * 512 + i] = temp[i * 4 + j][k];
}

//-------------------------------------
// Group 0b, Normalization
//-------------------------------------
void magika_group0b(float xy[256 * 512], float g[512], float b[512]) {
  // loop for 256 vectors
  for (int i = 0; i < 256; i++) {

    float *px = xy + (i * 512);

    // compute the mean
    float mean = 0;
    for (int j = 0; j < 512; j++)
      mean += px[j];
    mean = mean / 512.0;

    // compute the varance
    float var = 0;
    for (int j = 0; j < 512; j++)
      var += (px[j] - mean) * (px[j] - mean);
    var = var / 512.0;

    // normalize
    float scaling = 1.0 / sqrt(var + 9.999999974752427e-7);
    for (int j = 0; j < 512; j++)
      px[j] = b[j] + g[j] * scaling * (px[j] - mean);
  }
}

//------------------------------------------------
// Group 1, Conv 1D
// input is 512 pixels x 256 channels
//------------------------------------------------
float magika_gelu_approx(float x) {
  float y =
      (x / 2.0) *
      (1.0 + tanh(0.7978845834732056 * (x + 0.044714998453855515 * x * x * x)));
  return y;
}

void magika_group1(float x[256 * 512], float w[512 * 256 * 5], float b[512],
                   float y[512]) {
  // loop for 512 output channels
  for (int i = 0; i < 512; i++) {

    float ymax = -100;

    // loop for 508 output pixels
    for (int j = 0; j < 508; j++) {

      float sum = b[i];

      for (int k = 0; k < 256; k++)   // kth input channel
        for (int m = 0; m < 5; m++) { // m th tap
          float this_c = w[i + 512 * k + 512 * 256 * m];
          float this_x = x[j + m + 512 * k];
          sum += this_c * this_x;
        }

      // approximated gelu
      sum = magika_gelu_approx(sum);

      // find the max
      if ((j == 0) || (sum > ymax))
        ymax = sum;
    }

    // save ymax to output
    y[i] = ymax;
  }
}

//------------------------------------------------
// Group 2, MMULT + Softmax
//------------------------------------------------
void magika_group2(float x[512], float w[214 * 512], float b[214],
                   float y[214]) {

  // normalize x
  float mean = 0;
  for (int i = 0; i < 512; i++)
    mean += x[i];
  mean = mean / 512.0;

  // compute the varance
  float var = 0;
  for (int j = 0; j < 512; j++)
    var += (x[j] - mean) * (x[j] - mean);
  var = var / 512.0;

  // normalize
  float scaling = 1.0 / sqrt(var + 9.999999974752427e-7);
  for (int j = 0; j < 512; j++)
    x[j] = scaling * (x[j] - mean);

  // matrix multiplication
  for (int i = 0; i < 214; i++) {

    float sum = b[i];

    for (int k = 0; k < 512; k++) {
      float this_c = w[k + i * 512];
      float this_x = x[k];
      sum += this_c * this_x;
    }

    // save ymax to output
    y[i] = sum;
  }

  // softmax
  float ymax = y[0];
  for (int i = 1; i < 214; i++)
    if (y[i] > ymax)
      ymax = y[i];

  float ysum = 0;
  for (int i = 0; i < 214; i++) {
    y[i] = exp(y[i] - ymax);
    ysum += y[i];
  }

  ysum = 1.0 / ysum;
  for (int i = 0; i < 214; i++) {
    y[i] *= ysum;
  }
}

#ifdef MAGIKA_TOP

#include "magika_v3p3_weights.h"

//----------------------------------------------
// Magika v3.3 Model
//----------------------------------------------
char magika_v3p3(short xin[2048], float yout[214]) {

  // input range check
  char is_err = 0;
  for (int i = 0; i < 2048; i++) {
    int idx = xin[i];
    if (idx < 0 || idx > 256) {
      is_err = 1;
      printf("Magika_v3p3: Found invalid xin[%d]=%d\n", i, idx);
      break;
    }
  }
  if (is_err == 1)
    return -1;

  // Group 0
  float y0[256 * 512];
  magika_group0a(xin, magika_0a_c, y0);
  magika_group0b(y0, magika_0b_g, magika_0b_b);

  // Group 1
  float y1[512];
  magika_group1(y0, magika_1_w, magika_1_b, y1);

  // Group 2
  magika_group2(y1, magika_2_w, magika_2_b, yout);

  return 0;
}

#endif
