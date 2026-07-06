// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: LicenseRef-AMD-Proprietary

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//-----------------------------------------
// test FFTz model
//-----------------------------------------
int main(int argc, char *argv[]) {

  if (argc != 3) {
    printf("\n\tUsage: get_sqnr( ref_file_name, dut_file_name )\n\n");
    exit(-1);
  }

  FILE *fpin0 = fopen(argv[1], "rt");
  FILE *fpin1 = fopen(argv[2], "rt");

  if (fpin0 == NULL) {
    printf(" Error! Unable to open the ref file = %s\n", argv[1]);
    exit(-1);
  }

  if (fpin1 == NULL) {
    printf(" Error! Unable to open the dut file = %s\n", argv[2]);
    exit(-1);
  }

  float ref;
  double sumpwr = 0;
  double sumdif = 0;
  int din;

  int dut_int;
  float *dut_p = (float *)&dut_int;

  long long samplecnt = 0;

  printf("Computing EVM for %s against %s ...", argv[2], argv[1]);

  while (fscanf(fpin0, "%f", &ref) == 1) {

    samplecnt += 1;

    if (fscanf(fpin1, "%d", &din) == 1) {

      dut_int = din;

      float dut = *dut_p;

      double a = ref - dut;
      double b = ref;
      sumpwr += b * b;
      sumdif += a * a;
    } else {
      printf("\n\nWarning: DUT file is shorten than ref. Sample Cnt = %lld.\n",
             samplecnt);
      break;
    }
  }

  if (sumdif == 0) {
    printf("Input Files Bit-true Match.\n");
  } else {
    // printf("sum diff = %e, sum_pwr = %e\n", sumdif, sumpwr);
    printf("MSE = %.1f dBc\n", 10 * log10(sumdif / sumpwr));
  }

  fclose(fpin0);
  fclose(fpin1);
}
