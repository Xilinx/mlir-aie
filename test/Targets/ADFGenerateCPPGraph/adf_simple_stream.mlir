//===- adf_simple_stream.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate --adf-generate-cpp-graph %s | FileCheck %s

// CHECK:       #ifndef FUNCTION_KERNELS_H
// CHECK:       #define FUNCTION_KERNELS_H
// CHECK:       void kfunc1(input_stream_int32 * in0, output_stream_int32 * out0);
// CHECK:       void kfunc2(input_stream_int32 * in0, input_stream_int32 * in1, output_stream_int32 * out0);
// CHECK:       #endif

// CHECK:       #include <adf.h>
// CHECK:       using namespace adf;
// CHECK:       class simpleStream : public graph {
// CHECK:       private:
// CHECK:         kernel k1;
// CHECK:         kernel k2;
// CHECK:         kernel k3;
// CHECK:       public:
// CHECK:         input_port gin;
// CHECK:         output_port gout;
// CHECK:         simpleStream() {
// CHECK:           k1 = kernel::create(kfunc1);
// CHECK:           k2 = kernel::create(kfunc2);
// CHECK:           k3 = kernel::create(kfunc2);
// CHECK:           connect<stream> n0 (gin, k3.in[0]);
// CHECK:           connect<stream> n1 (gin, k2.in[0]);
// CHECK:           connect<stream> n2 (gin, k1.in[0]);
// CHECK:           connect<stream> n3 (k1.out[0], k2.in[1]);
// CHECK:           connect<stream> n4 (k2.out[0], k3.in[1]);
// CHECK:           connect<stream> n5 (k3.out[0], gout);
// CHECK:         }
// CHECK:       }

// "blk" is block, it could be window or stream form in ADF graph.
module {
    func.func private @kfunc1(%in1 : !ADF.stream<!ADF.int32>)
                             ->(!ADF.stream<!ADF.int32>)

    func.func private @kfunc2(%in1 : !ADF.stream<!ADF.int32>,
                         %in2 : !ADF.stream<!ADF.int32>)
                             ->(!ADF.stream<!ADF.int32>)

    ADF.graph("simpleStream") {
        %gi = ADF.input_port("gin")  [1:i1, -1:i32] -> !ADF.interface<!ADF.int32>
        %2 = ADF.kernel @kfunc1(%gi) : (!ADF.interface<!ADF.int32>) -> !ADF.interface<!ADF.int32>
        %3 = ADF.kernel @kfunc2(%gi, %2) : (!ADF.interface<!ADF.int32>, !ADF.interface<!ADF.int32>) -> !ADF.interface<!ADF.int32>
        %4 = ADF.kernel @kfunc2(%gi, %3) : (!ADF.interface<!ADF.int32>, !ADF.interface<!ADF.int32>) -> !ADF.interface<!ADF.int32>
        %go = ADF.output_port("gout") %4 : (!ADF.interface<!ADF.int32>) -> !ADF.interface<!ADF.int32>
    }
}


// --------------------------------------------------------------------------------
// target C++ code
// ----------------------- simpleStreamGraph.cpp ---------------------------------------

// #include <adf.h>
// #include "kernels.h"

// using namespace adf;

// class simpleGraph : public graph {
// private:
//   kernel k1;
//   kernel k2;
//   kernel k3;
// public:
//   input_port  gin;    // default graph input name
//   output_port gout;   // default graph output name

//   simpleGraph() {
//     k1 = kernel::create(kfunc1);
//     k2 = kernel::create(kfunc2);
//     k3 = kernel::create(kfunc3);

//     connect< window<128>> net0 (gin,       k1.in[0]);
//     connect< window<128>> net1 (gin,       k2.in[0]);
//     connect< window<128>> net2 (k1.out[0], k2.in[1]);
//     connect< window<128>> net3 (gin,       k3.in[0]);
//     connect< window<128>> net4 (k2.out[0], k3.in[1]);
//     connect< window<128>> net4 (k3.out[0], gout);

//   }
// }
