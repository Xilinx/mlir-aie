// RUN: aie-translate %s -mlir-to-llvmir | FileCheck %s

// ----- MAC ----- 

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

// ----- MUL ----- 

// CHECK-LABEL: define <16 x i64> @mul_conf_acc32
llvm.func @mul_conf_acc32(%A : vector<64xi8>,
                          %B : vector<16xi32>,
                          %cfg : i32)
                          -> vector<16xi64> {
    // CHECK: call <16 x i64> @llvm.aie2.I512.I512.acc32.mul.conf(
    // CHECK-SAME: <64 x i8> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I512.I512.acc32.mul.conf"(%A, %B, %cfg) : 
        (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
    llvm.return %0 : vector<16xi64>
}

// CHECK-LABEL: define <16 x i64> @mul_conf_acc64
llvm.func @mul_conf_acc64(%A : vector<64xi8>,
                          %B : vector<16xi32>,
                          %cfg : i32)
                          -> vector<16xi64> {
    // CHECK: call <16 x i64> @llvm.aie2.I512.I512.acc64.mul.conf(
    // CHECK-SAME: <64 x i8> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I512.I512.acc64.mul.conf"(%A, %B, %cfg) :
        (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
    llvm.return %0 : vector<16xi64>
}

// CHECK-LABEL: define <8 x i64> @mul_conf_bf16
llvm.func @mul_conf_bf16(%A : vector<32xbf16>,
                         %B : vector<32xbf16>,
                         %cfg : i32)
                         -> vector<8xi64> {
    // CHECK: call <8 x i64> @llvm.aie2.bf.mul16.conf(
    // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.bf.mul16.conf"(%A, %B, %cfg) : 
        (vector<32xbf16>, vector<32xbf16>, i32) -> vector<8xi64>
    llvm.return %0 : vector<8xi64>
}

// ----- SET ----- 

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

// ----- SRS ----- 

// CHECK-LABEL: define <32 x i16> @srs_512b_v32_acc32
llvm.func @srs_512b_v32_acc32(%v : vector<16xi64>, %shft : i32, %sign : i32) -> vector<32xi16> {
    // CHECK: call <32 x i16> @llvm.aie2.I512.v32.acc32.srs(
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I512.v32.acc32.srs"(%v, %shft, %sign) :
                                        (vector<16xi64>, i32, i32) -> vector<32xi16>
    llvm.return %0 : vector<32xi16>
}

// CHECK-LABEL: define <32 x i8> @srs_256b_v32_acc32
llvm.func @srs_256b_v32_acc32(%v : vector<16xi64>, %shft : i32, %sign : i32) -> vector<32xi8> {
    // CHECK: call <32 x i8> @llvm.aie2.I256.v32.acc32.srs(
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I256.v32.acc32.srs"(%v, %shft, %sign) :
                                        (vector<16xi64>, i32, i32) -> vector<32xi8>
    llvm.return %0 : vector<32xi8>
}

// CHECK-LABEL: define <16 x i32> @srs_512b_v16_acc64
llvm.func @srs_512b_v16_acc64(%v : vector<16xi64>, %shft : i32, %sign : i32) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.I512.v16.acc64.srs(
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I512.v16.acc64.srs"(%v, %shft, %sign) : 
                                        (vector<16xi64>, i32, i32) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// CHECK-LABEL: define <16 x bfloat> @srs_256b_v16_accfloat
llvm.func @srs_256b_v16_accfloat(%v : vector<8xi64>) -> vector<16xbf16> {
    // CHECK: call <16 x bfloat> @llvm.aie2.v16accfloat.to.v16bf16(
    // CHECK-SAME: <8 x i64> %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%v) : (vector<8xi64>) -> vector<16xbf16>
    llvm.return %0 : vector<16xbf16>
}

// ----- BROADCAST ----- 

// CHECK-LABEL: define <64 x i8> @vbroadcast8_i512
llvm.func @vbroadcast8_i512(%val : i32) -> vector<64xi8> {
    // CHECK: call <64 x i8> @llvm.aie2.vbroadcast8.I512(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vbroadcast8.I512"(%val) : (i32) -> vector<64xi8>
    llvm.return %0 : vector<64xi8>
}

// CHECK-LABEL: define <16 x i32> @vbroadcast32_i512
llvm.func @vbroadcast32_i512(%val : i32) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.vbroadcast32.I512(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vbroadcast32.I512"(%val) : (i32) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// CHECK-LABEL: define <32 x bfloat> @vbroadcast16_bf512
llvm.func @vbroadcast16_bf512(%val : bf16) -> vector<32xbf16> {
    // CHECK: call <32 x bfloat> @llvm.aie2.vbroadcast16.bf512(
    // CHECK-SAME: bfloat %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vbroadcast16.bf512"(%val) : (bf16) -> vector<32xbf16>
    llvm.return %0 : vector<32xbf16>
}

// ----- EXT ----- 

// CHECK-LABEL: define <8 x i32> @ext_i256_i512
llvm.func @ext_i256_i512(%v : vector<16xi32>, %idx : i32) -> vector<8xi32> {
    // CHECK: call <8 x i32> @llvm.aie2.ext.I256.I512(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %1 = "xllvm.intr.aie2.ext.I256.I512"(%v, %idx) : 
                                        (vector<16xi32>, i32) -> vector<8xi32>
    llvm.return %1 : vector<8xi32>
}

// ----- CONCAT ----- 

// CHECK-LABEL: define <16 x i32> @concat_i512_i256
llvm.func @concat_i512_i256(%a : vector<8xi32>, %b : vector<8xi32>) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.concat.I512.I256(
    // CHECK-SAME: <8 x i32> %{{[0-9]+}}, <8 x i32> %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.concat.I512.I256"(%a, %b) : 
                                        (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// ----- SHUFFLE ----- 

// CHECK-LABEL: define <16 x i32> @shuffle_i512
llvm.func @shuffle_i512(%a : vector<16xi32>, %b : vector<16xi32>, , %mode : i32) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.vshuffle(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vshuffle"(%a, %b, %mode) : 
                                        (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// ----- UNDEF ----- 

// CHECK-LABEL: define <16 x i32> @undef_v16i32
llvm.func @undef_v16i32() -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.v16int32(
    %0 ="xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}
