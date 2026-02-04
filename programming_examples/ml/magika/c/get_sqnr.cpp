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
