// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

// This file contains symbols that are used in Chess compilation to implement
// intrinsics.  It tries to solve the problem that the Chess compiler does not
// implement 'standard' llvm intrinsics (which contain '.') as a separator.
// Rather than implement a fragile mechanism that relies on a particular encoding
// this file is compiled with the Chess compiler to generate the proper encoding
// and then this file is linked with code generated with llvm-style intrinsics.
// Note that the Chess frontend replaces '.' with '___' when parsing .ll code
// containing standard intrinsic names, so these symbols are defined that way.

extern "C" void llvm___aie___lock___acquire___reg(unsigned id, unsigned val) {
    acquire(id, val);
}
extern "C" void llvm___aie___lock___release___reg(unsigned id, unsigned val) {
    release(id, val);
}