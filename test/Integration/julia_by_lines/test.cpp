//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <xaiengine.h>

#include "kernel.h"
#include <time.h>

using namespace std;

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 1, 3);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  int errors = 0;

  // overall size of the frame.
  int framerows = 1024;
  int framecols = 1024;
  std::vector<int32_t> frame(framerows * framecols);

  printf("-- Acquire lock first.\n");
  mlir_aie_acquire_lock(_xaie, 1, 3, 3, 0, 0); // Should this part of setup???

  printf("-- Start cores\n");
  mlir_aie_start_cores(_xaie);

  mlir_aie_check_float("Before release lock:",
                       mlir_aie_read_buffer_debuf(_xaie, 5), 0.0f, errors);

  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  struct timespec computedone[framerows];
  struct timespec copydone[framerows];
  for (int i = 0; i < framerows; i++) {

    //        printf("-- Release lock.\n");
    mlir_aie_release_lock(_xaie, 1, 3, 3, 1, 1);

    int tries = 1;
    //        printf("Waiting to acquire lock ...\n");
    while (tries < 1000 && !mlir_aie_acquire_lock(_xaie, 1, 3, 3, 0, 1)) {
      tries++;
    }
    clock_gettime(CLOCK_REALTIME, &(computedone[i]));
    if (tries >= 1000) {
      printf("It took %d tries.\n", tries);
      exit(1);
    }

    for (int c = 0; c < framecols; c++) {
      frame[framecols * i + c] = mlir_aie_read_buffer_a(_xaie, c);
    }
    clock_gettime(CLOCK_REALTIME, &(copydone[i]));
  }
  struct timespec end;
  clock_gettime(CLOCK_REALTIME, &end);
  double t = (end.tv_sec - start.tv_sec);
  t += (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0f;

  double computetime = 0;
  double copytime = 0;
  for (int i = 1; i < framerows; i++) {
    computetime += (computedone[i].tv_sec - copydone[i - 1].tv_sec);
    computetime += (double)(computedone[i].tv_nsec - copydone[i - 1].tv_nsec) /
                   1000000000.0f;
    copytime += (copydone[i].tv_sec - computedone[i].tv_sec);
    copytime +=
        (double)(copydone[i].tv_nsec - computedone[i].tv_nsec) / 1000000000.0f;
  }

  clock_gettime(CLOCK_REALTIME, &start);

  float MinRe = -1.5;
  float MaxRe = 0.5;
  float MinIm = -1.0;
  float MaxIm = 1.0;
  float StepRe = (MaxRe - MinRe) / framecols;
  float StepIm = (MaxIm - MinIm) / framerows;
  float Im = MinIm;
  int32_t line[framecols];
  for (int i = 0; i < framerows; i++) {
    do_line(line, MinRe, StepRe, Im, framecols);
    Im += StepIm;
  }
  clock_gettime(CLOCK_REALTIME, &end);
  double t2 = (end.tv_sec - start.tv_sec);
  t2 += (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0f;

  printf("%f ARM compute seconds \n", t2);
  printf("%f AIE seconds \n", t);
  printf("  %f compute seconds \n", computetime);
  printf("  %f copy seconds \n", copytime);

  ofstream myfile;
  myfile.open("julia.pgm");
  myfile << "P2\n";                                // Magic
  myfile << framecols << " " << framerows << "\n"; // Width, height
  myfile << "255\n";                               // max value

  for (int i = 0; i < framerows; i++) {
    for (int j = 0; j < framecols; j++) {
      unsigned t = frame[framecols * i + j];
      // if(t > 40) printf("#");
      // else if(t > 20) printf("*");
      // else if(t > 8) printf(".");
      // else printf(" ");
      myfile << t << " ";
    }
    // printf("\n");
    myfile << "\n";
  }
  myfile.close();

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }
  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");
  return res;
}
