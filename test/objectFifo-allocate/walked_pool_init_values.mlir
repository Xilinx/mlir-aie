// RUN: aie-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --aie-objectFifo-stateful-transform %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// @y's 30 objects and @z's link need 50 BD ids, so @y is walked whole: one
// buffer of 30 objects, which starts holding every object's initial value in
// object order, splat or not.
// CHECK-LABEL: @walked_init_i16
// CHECK: aie.buffer(%{{.*}}) {sym_name = "y__buf0"} : memref<60xi16> = dense<[1, -1, 2, 2, 3, -3, 4, 4, 5, -5, 6, 6, 7, -7, 8, 8, 9, -9, 10, 10, 11, -11, 12, 12, 13, -13, 14, 14, 15, -15, 16, 16, 17, -17, 18, 18, 19, -19, 20, 20, 21, -21, 22, 22, 23, -23, 24, 24, 25, -25, 26, 26, 27, -27, 28, 28, 29, -29, 30, 30]>
// CHECK-LABEL: @walked_init_f32
// CHECK: aie.buffer(%{{.*}}) {sym_name = "y__buf0"} : memref<60xf32> = dense<[1.500000e+00, -1.250000e+00, 2.000000e+00, 2.000000e+00, 3.500000e+00, -3.250000e+00, 4.000000e+00, 4.000000e+00, 5.500000e+00, -5.250000e+00, 6.000000e+00, 6.000000e+00, 7.500000e+00, -7.250000e+00, 8.000000e+00, 8.000000e+00, 9.500000e+00, -9.250000e+00, 1.000000e+01, 1.000000e+01, 1.150000e+01, -1.125000e+01, 1.200000e+01, 1.200000e+01, 1.350000e+01, -1.325000e+01, 1.400000e+01, 1.400000e+01, 1.550000e+01, -1.525000e+01, 1.600000e+01, 1.600000e+01, 1.750000e+01, -1.725000e+01, 1.800000e+01, 1.800000e+01, 1.950000e+01, -1.925000e+01, 2.000000e+01, 2.000000e+01, 2.150000e+01, -2.125000e+01, 2.200000e+01, 2.200000e+01, 2.350000e+01, -2.325000e+01, 2.400000e+01, 2.400000e+01, 2.550000e+01, -2.525000e+01, 2.600000e+01, 2.600000e+01, 2.750000e+01, -2.725000e+01, 2.800000e+01, 2.800000e+01, 2.950000e+01, -2.925000e+01, 3.000000e+01, 3.000000e+01]>

module @walked_init_i16 {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    aie.objectfifo @y(%m, {%s}, 30 : i32) : !aie.objectfifo<memref<2xi16>> = [dense<[1, -1]> : memref<2xi16>, dense<2> : memref<2xi16>, dense<[3, -3]> : memref<2xi16>, dense<4> : memref<2xi16>, dense<[5, -5]> : memref<2xi16>, dense<6> : memref<2xi16>, dense<[7, -7]> : memref<2xi16>, dense<8> : memref<2xi16>, dense<[9, -9]> : memref<2xi16>, dense<10> : memref<2xi16>, dense<[11, -11]> : memref<2xi16>, dense<12> : memref<2xi16>, dense<[13, -13]> : memref<2xi16>, dense<14> : memref<2xi16>, dense<[15, -15]> : memref<2xi16>, dense<16> : memref<2xi16>, dense<[17, -17]> : memref<2xi16>, dense<18> : memref<2xi16>, dense<[19, -19]> : memref<2xi16>, dense<20> : memref<2xi16>, dense<[21, -21]> : memref<2xi16>, dense<22> : memref<2xi16>, dense<[23, -23]> : memref<2xi16>, dense<24> : memref<2xi16>, dense<[25, -25]> : memref<2xi16>, dense<26> : memref<2xi16>, dense<[27, -27]> : memref<2xi16>, dense<28> : memref<2xi16>, dense<[29, -29]> : memref<2xi16>, dense<30> : memref<2xi16>]
    aie.objectfifo @z(%s, {%m}, 10 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @z_out(%m, {%s}, 10 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@z] -> [@z_out] ([] [])
  }
}

// -----

module @walked_init_f32 {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    aie.objectfifo @y(%m, {%s}, 30 : i32) : !aie.objectfifo<memref<2xf32>> = [dense<[1.5, -1.25]> : memref<2xf32>, dense<2.0> : memref<2xf32>, dense<[3.5, -3.25]> : memref<2xf32>, dense<4.0> : memref<2xf32>, dense<[5.5, -5.25]> : memref<2xf32>, dense<6.0> : memref<2xf32>, dense<[7.5, -7.25]> : memref<2xf32>, dense<8.0> : memref<2xf32>, dense<[9.5, -9.25]> : memref<2xf32>, dense<10.0> : memref<2xf32>, dense<[11.5, -11.25]> : memref<2xf32>, dense<12.0> : memref<2xf32>, dense<[13.5, -13.25]> : memref<2xf32>, dense<14.0> : memref<2xf32>, dense<[15.5, -15.25]> : memref<2xf32>, dense<16.0> : memref<2xf32>, dense<[17.5, -17.25]> : memref<2xf32>, dense<18.0> : memref<2xf32>, dense<[19.5, -19.25]> : memref<2xf32>, dense<20.0> : memref<2xf32>, dense<[21.5, -21.25]> : memref<2xf32>, dense<22.0> : memref<2xf32>, dense<[23.5, -23.25]> : memref<2xf32>, dense<24.0> : memref<2xf32>, dense<[25.5, -25.25]> : memref<2xf32>, dense<26.0> : memref<2xf32>, dense<[27.5, -27.25]> : memref<2xf32>, dense<28.0> : memref<2xf32>, dense<[29.5, -29.25]> : memref<2xf32>, dense<30.0> : memref<2xf32>]
    aie.objectfifo @z(%s, {%m}, 10 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @z_out(%m, {%s}, 10 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@z] -> [@z_out] ([] [])
  }
}
