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
#include <time.h>

#include "magika_v3p3_weights.h"

// char magika_v3p3(short xin[2048], float yout[256 * 512]);
void magika_group0a(short xin[2048], float lkup[257 * 64],
                    float yout[256 * 512]);
void magika_group0b(float xy[256 * 512], float g[512], float b[512]);
void magika_group1(float x[256 * 512], float w[512 * 256 * 5], float b[512],
                   float y[512]);
void magika_group2(float x[512], float w[214 * 512], float b[214],
                   float y[214]);

double get_evm(unsigned int len, float *gold, float *dut) {
  double diff_pwr = 0;
  double tota_pwr = 0;

  for (unsigned int i = 0; i < len; i++) {

    double diff = gold[i] - dut[i];

    diff_pwr += diff * diff;
    tota_pwr += gold[i] * gold[i];
  }

  double evm = 10 * log10(diff_pwr / tota_pwr);

  return evm;
}

//-----------------------------------------
// test dsa key generator
//-----------------------------------------
int main() {

  FILE *fpd = fopen("../data/din.txt", "rt");
  FILE *fxin = fopen("../data/fxin.txt", "wt");

  FILE *g0a = fopen("../data/g0a.txt", "wt");
  FILE *g0b = fopen("../data/g0b.txt", "wt");
  FILE *g1 = fopen("../data/g1.txt", "wt");
  FILE *g2 = fopen("../data/g2.txt", "wt");

  if (fpd == NULL) {
    printf(" Error! Unable to open the input file\n");
    exit(-1);
  }

  FILE *fpr = fopen("../data/ref.txt", "rt");
  if (fpr == NULL) {
    printf(" Error! Unable to open the ref file\n");
    exit(-1);
  }

  if (fxin == NULL) {
    printf(" Error! Unable to open the fxin file\n");
    exit(-1);
  }

  if (g0a == NULL) {
    printf(" Error! Unable to open the g0a file\n");
    exit(-1);
  }

  if (g0b == NULL) {
    printf(" Error! Unable to open the g0b file\n");
    exit(-1);
  }

  if (g1 == NULL) {
    printf(" Error! Unable to open the g1 file\n");
    exit(-1);
  }

  if (g2 == NULL) {
    printf(" Error! Unable to open the g2 file\n");
    exit(-1);
  }

  short xin[2048];
  float ref[214];
  float yout[214];

  int n_ite = 1;

  printf("\n\n-----------------------------------\n");
  printf("--  Magika C Model Verification  --\n");
  printf("-----------------------------------\n");

  while (1) {

    // read inputs
    printf("Reading inputs\n");
    char is_valid = 1;
    for (int i = 0; i < 2048; i++) {

      int din;
      int ncnt = fscanf(fpd, "%d", &din);

      if (ncnt == 1) {
        xin[i] = din;
      } else {
        is_valid = 0;
        break;
      }
    }
    if (is_valid != 1)
      break;

    // read golden ref
    printf("Reading golden ref\n");
    for (int i = 0; i < 214; i++) {
      float din;
      int ncnt = fscanf(fpr, "%f", &din);

      if (ncnt == 1) {
        ref[i] = din;
      } else {
        is_valid = 0;
        break;
      }
    }
    if (is_valid != 1)
      break;

    // call dut
    // char stat = magika_v3p3(xin, yout);

    for (int i = 0; i < 2048; i++) {
      fprintf(fxin, "%d\n", xin[i]);
    }

    // input range check
    printf("Input range check\n");
    char is_err = 0;
    for (int i = 0; i < 2048; i++) {
      int idx = xin[i];
      if (idx < 0 || idx > 256) {
        is_err = 1;
        printf("Magika_v3p3: Found invalid xin[%d]=%d\n", i, idx);
        break;
      }
    }
    if (is_err == 1) {
      printf("Magika_v3p3: Error. Invalid din input. Exiting.\n");
      return -1;
    }

    // Group 0
    printf("Group0a\n");
    float y0[256 * 512];
    magika_group0a(xin, magika_0a_c, y0);

    for (int i = 0; i < 256 * 512; i++) {
      fprintf(g0a, "%E\n", y0[i]);
    }

    printf("Group0b\n");
    magika_group0b(y0, magika_0b_g, magika_0b_b);

    for (int i = 0; i < 256 * 512; i++) {
      fprintf(g0b, "%E\n", y0[i]);
    }

    // Group 1
    float y1[512];
    printf("Group1\n");
    magika_group1(y0, magika_1_w, magika_1_b, y1);

    for (int i = 0; i < 512; i++) {
      fprintf(g1, "%E\n", y1[i]);
    }

    // Group 2
    printf("Group2\n");
    magika_group2(y1, magika_2_w, magika_2_b, yout);

    for (int i = 0; i < 214; i++) {
      fprintf(g2, "%E\n", yout[i]);
    }

    /*
        if (stat < 0) {
          printf("Error: Magika v3p3 Stat = %d.\n", stat);
          break;
        }
    */

    double this_evm = get_evm(214, ref, yout);

    printf(" ITE = %3d : EVM = %f dBc\n", n_ite++, this_evm);
  }

  printf("\n\nTotal Test = %d.\n\n", n_ite - 1);
  fclose(fpd);
  fclose(fpr);
  fclose(g0a);
  fclose(g0b);
  fclose(g1);
  fclose(g2);
}
