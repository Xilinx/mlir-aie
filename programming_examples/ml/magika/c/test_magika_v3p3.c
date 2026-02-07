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

char magika_v3p3(short xin[2048], float yout[256 * 512]);

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

  if (fpd == NULL) {
    printf(" Error! Unable to open the input file\n");
    exit(-1);
  }

  FILE *fpr = fopen("../data/ref.txt", "rt");
  if (fpr == NULL) {
    printf(" Error! Unable to open the ref file\n");
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
    char stat = magika_v3p3(xin, yout);

    if (stat < 0) {
      printf("Error: Magika v3p3 Stat = %d.\n", stat);
      break;
    }

    double this_evm = get_evm(214, ref, yout);

    printf(" ITE = %3d : EVM = %f dBc\n", n_ite++, this_evm);
  }

  printf("\n\nTotal Test = %d.\n\n", n_ite - 1);
  fclose(fpd);
  fclose(fpr);
}
