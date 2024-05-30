//===- adf_simple_window.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate --adf-generate-cpp-graph %s | tee %T/graph.h | FileCheck %s
// UN: source /proj/xbuilds/2021.1_daily_latest/installs/lin64/Vitis/2021.1/settings64.sh; /proj/xbuilds/2021.1_daily_latest/installs/lin64/Vitis/2021.1/aietools/bin/aiecompiler %S/project_window.cpp -I %T

// CHECK:       #ifndef FUNCTION_KERNELS_H
// CHECK:       #define FUNCTION_KERNELS_H
// CHECK:       void kfunc1(input_window_int32 * in0, output_window_int16 * out0);
// CHECK:       void kfunc2(int32 in0, input_window_int16 * in1, output_window_int32 * out0);
// CHECK:       void kfunc3(input_window_int32 * in0, input_window_int32 * in1, output_window_int32 * out0);
// CHECK:       #endif

// CHECK:       #include <adf.h>
// CHECK:       using namespace adf;
// CHECK:       class simpleWindow : public graph {
// CHECK:       private:
// CHECK:         kernel k1;
// CHECK:         kernel k2;
// CHECK:         kernel k3;
// CHECK:       public:
// CHECK:         input_port gin;
// CHECK:         input_port gp;
// CHECK:         output_port gout;
// CHECK:         simpleWindow() {
// CHECK:           k1 = kernel::create(kfunc1);
// CHECK:           k2 = kernel::create(kfunc2);
// CHECK:           k3 = kernel::create(kfunc3);
// CHECK:           connect<window<128> > n0 (gin, k3.in[0]);
// CHECK:           connect<window<128> > n1 (gin, k1.in[0]);
// CHECK:           connect<parameter> n2 (gp, k2.in[0]);
// CHECK:           connect<window<128> > n3 (k1.out[0], k2.in[1]);
// CHECK:           connect<window<64> > n4 (k2.out[0], k3.in[1]);
// CHECK:           connect<window<128> > n5 (k3.out[0], gout);
// CHECK:         }
// CHECK:       }

// "blk" is block, it could be window or stream form in ADF graph.
module {
    func.func private @kfunc1(%in1 : !ADF.window<!ADF.int32, 128, 0>)
                             ->(!ADF.window<!ADF.int16, 128, 0>)
    func.func private @kfunc2(%in1 : !ADF.parameter<!ADF.int32>,
                         %in2 : !ADF.window<!ADF.int16, 128, 0>)
                             ->(!ADF.window<!ADF.int32, 64, 0>)
    func.func private @kfunc3(%in1 : !ADF.window<!ADF.int32, 128, 0>,
                         %in2 : !ADF.window<!ADF.int32, 64, 0>)
                             ->(!ADF.window<!ADF.int32, 128, 0>)

    ADF.graph("simpleWindow") {
        %gi = ADF.input_port("gin")  [0:i1, 128:i32] -> !ADF.interface<!ADF.int32> 
        %gp = ADF.input_port("gp")  [0:i1, 128:i32] -> !ADF.interface<!ADF.int32> 
        %2 = ADF.kernel @kfunc1(%gi) : (!ADF.interface<!ADF.int32> ) -> !ADF.interface<!ADF.int16> 
        %3 = ADF.kernel @kfunc2(%gp, %2) : (!ADF.interface<!ADF.int32> , !ADF.interface<!ADF.int16> ) -> !ADF.interface<!ADF.int32> 
        %4 = ADF.kernel @kfunc3(%gi, %3) : (!ADF.interface<!ADF.int32> , !ADF.interface<!ADF.int32> ) -> !ADF.interface<!ADF.int32> 
        %go = ADF.output_port("gout") %4 : (!ADF.interface<!ADF.int32> ) -> !ADF.interface<!ADF.int32> 
    }
}


// --------------------------------------------------------------------------------
// target C++ code
// ----------------------- simpleWindowGraph.cpp ---------------------------------------

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
