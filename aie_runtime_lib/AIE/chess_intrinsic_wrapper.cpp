//===- chess_intrinsic_wrapper.cpp ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

/// \file
/// This file contains symbols that are used in Chess compilation to implement
/// intrinsics.  It tries to solve the problem that the Chess compiler does not
/// implement 'standard' llvm intrinsics (which contain '.') as a separator.
/// Rather than implement a fragile mechanism that relies on a particular
/// encoding this file is compiled with the Chess compiler to generate the
/// proper encoding and then this file is linked with code generated with
/// llvm-style intrinsics. Note that the Chess frontend replaces '.' with '___'
/// when parsing .ll code containing standard intrinsic names, so these symbols
/// are defined that way.

extern "C" void llvm___aie___lock___acquire___reg(unsigned id, unsigned val) {
  acquire(id, val);
}
extern "C" void llvm___aie___lock___release___reg(unsigned id, unsigned val) {
  release(id, val);
}

extern "C" int llvm___aie___get___ss(int stream) { return get_ss(stream); }
extern "C" float llvm___aie___getf___ss(int stream) { return getf_ss(stream); }
extern "C" int llvm___aie___get___ss0___tlast() { return get_ss0_tlast(); }
extern "C" int llvm___aie___get___ss1___tlast() { return get_ss1_tlast(); }
extern "C" int llvm___aie___get___wss0___tlast() { return get_wss0_tlast(); }
extern "C" int llvm___aie___get___wss1___tlast() { return get_wss1_tlast(); }
extern "C" void llvm___aie___put___ms(int idx_ms, int a) { put_ms(idx_ms, a); }
//extern "C" void llvm___aie___put___ms___tlast(int idx_ms, int a, int tlast) { put_ms(idx_ms, a, tlast); }