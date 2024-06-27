// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

typedef int16_t my_t;

extern "C" {
void concat(my_t *a, my_t *b, my_t *c, int a_sz, int b_sz, int c_sz) {
  // Concatenates a and b and writes the result to c.
  int i = 0;
  for (; i < c_sz && i < a_sz; i++) {
    c[i] = a[i];
  }
  for (; i < c_sz && i - a_sz < b_sz; i++) {
    c[i] = b[i - a_sz];
  }
}
}
