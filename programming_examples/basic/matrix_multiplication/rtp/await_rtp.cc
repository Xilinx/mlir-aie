//===- await_rtp.cc ---------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AWAIT_RTP_CC
#define AWAIT_RTP_CC

extern "C" {
/* Polls a run-time parameter to be set to a value other than -1.
   
   Dedicate one RTP as the "ready" signal. Once you have set all other RTPs,
   set this RTP to 1, and this function will unblock. In the core, you can
   then read the other RTPs *after* this function unblocks.

   Note: There is a small race condition here if the host sets the "ready"
   RTP *before* the core calls this function. This is unlikely to happen if
   this function is the first thing called in the core, as the core 
   executes much faster than the host controller can set values in core 
   memory.
   */
void await_rtp(volatile int *rtp) {
  rtp[0] = -1;
  while(rtp[0] == -1);
}

int get_volatile_rtp(volatile int *rtp, int index) {
  return rtp[index];
}

}

#endif