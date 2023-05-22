//===- kernel13.cc ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
/*
extern "C" void do_mul(int32_t *buf)
{
    int tmp = buf[3];
    int val = tmp + 1;
    v8acc48 v8acc;
    v8int16 v8;
    v8 = upd_elem(v8,0,val);
    v8acc = ups(v8,0);
    put_mcd(v8acc);
}
*/
extern "C" void do_mul(int32_t *buf) {
  int tmp = buf[3];
  int val = tmp + tmp;
  val += tmp;
  val += tmp;
  val += tmp;
  v8acc48 v8acc;
  v8int16 v8;
  v8 = upd_elem(v8, 0, val);
  v8acc = ups(v8, 0);
  put_mcd(v8acc);
}
/*
extern "C" void do_mac(int32_t *buf)
{
    buf[0] = 17;
    int tmp = ext_elem(srs(get_scd(),0),0);
    buf[5] = tmp;
}
*/
extern "C" void do_mac(int32_t *buf) {
  int tmp = ext_elem(srs(get_scd(), 0), 0);
  int val = tmp + tmp;
  val += tmp;
  val += tmp;
  val += tmp;
  buf[5] = val;
}

#ifdef TEST
int32_t buf[32];

int main() {
  do_mul(buf);
  do_mac(buf);
  // printf("test is %d\n",buf[8]);
}
#endif
