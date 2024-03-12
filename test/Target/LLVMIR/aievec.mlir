// RUN: aie-translate %s -mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define <16 x i64> @mac_conf_acc32
llvm.func @mac_conf_acc32(%A : vector<64xi8>,
                          %B : vector<16xi32>,
                          %C : vector<16xi64>,
                          %cfg : i32)
                          -> vector<16xi64> {
    // CHECK: call <16 x i64> @llvm.aie2.I512.I512.ACC1024.acc32.mac.conf(
    // CHECK-SAME: <64 x i8> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}},
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(%A, %B, %C, %cfg) :
        (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
    llvm.return %0 : vector<16xi64>
}

// CHECK-LABEL: define <8 x i64> @mac_conf_bf16
llvm.func @mac_conf_bf16(%A : vector<32xbf16>,
                         %B : vector<32xbf16>,
                         %C : vector<8xi64>,
                         %cfg : i32)
                         -> vector<8xi64> {
    // CHECK: call <8 x i64> @llvm.aie2.bf.mac16.conf(
    // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}},
    // CHECK-SAME: <8 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.bf.mac16.conf"(%A, %B, %C, %cfg) :
        (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
    llvm.return %0 : vector<8xi64>
}

// CHECK-LABEL: define <16 x i32> @vector_set_128b_into_512b
llvm.func @vector_set_128b_into_512b(%v : vector<4xi32>) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.set.I512.I128(<4 x i32>
    %0 = "xllvm.intr.aie2.set.I512.I128"(%v) : (vector<4xi32>) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// CHECK-LABEL: define <16 x i32> @vector_set_256b_into_512b
llvm.func @vector_set_256b_into_512b(%v : vector<8xi32>) -> vector<16xi32> {
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: call <16 x i32> @llvm.aie2.set.I512.I256(<8 x i32>
    %1 = "xllvm.intr.aie2.set.I512.I256"(%v, %0) :
                                        (vector<8xi32>, i32) -> vector<16xi32>
    llvm.return %1 : vector<16xi32>
}
