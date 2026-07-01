// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: LicenseRef-AMD-Proprietary

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
