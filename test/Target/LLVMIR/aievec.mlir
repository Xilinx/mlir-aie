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

// ----- MSC ----- 

// CHECK-LABEL: define <8 x i64> @msc_conf_bf16
llvm.func @msc_conf_bf16(%A : vector<32xbf16>,
                         %B : vector<32xbf16>,
                         %Acc : vector<8xi64>,
                         %cfg : i32)
                         -> vector<8xi64> {
    // CHECK: call <8 x i64> @llvm.aie2.bf.msc16.conf(
    // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}},
    // CHECK-SAME: <8 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.bf.msc16.conf"(%A, %B, %Acc, %cfg) : 
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

// CHECK-LABEL: define <16 x i16> @srs_256b_v16_acc32
llvm.func @srs_256b_v16_acc32(%v : vector<8xi64>, %shft : i32, %sign : i32) -> vector<16xi16> {
    // CHECK: call <16 x i16> @llvm.aie2.I256.v16.acc32.srs(
    // CHECK-SAME: <8 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I256.v16.acc32.srs"(%v, %shft, %sign) :
                                        (vector<8xi64>, i32, i32) -> vector<16xi16>
    llvm.return %0 : vector<16xi16>
}

// CHECK-LABEL: define <16 x i16> @srs_256b_v16_acc64
llvm.func @srs_256b_v16_acc64(%v : vector<16xi64>, %shft : i32, %sign : i32) -> vector<16xi16> {
    // CHECK: call <16 x i16> @llvm.aie2.I256.v16.acc64.srs(
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I256.v16.acc64.srs"(%v, %shft, %sign) :
                                        (vector<16xi64>, i32, i32) -> vector<16xi16>
    llvm.return %0 : vector<16xi16>
}

// CHECK-LABEL: define <32 x i8> @srs_256b_v32_acc32
llvm.func @srs_256b_v32_acc32(%v : vector<16xi64>, %shft : i32, %sign : i32) -> vector<32xi8> {
    // CHECK: call <32 x i8> @llvm.aie2.I256.v32.acc32.srs(
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I256.v32.acc32.srs"(%v, %shft, %sign) :
                                        (vector<16xi64>, i32, i32) -> vector<32xi8>
    llvm.return %0 : vector<32xi8>
}

// CHECK-LABEL: define <8 x i32> @srs_256b_v8_acc64
llvm.func @srs_256b_v8_acc64(%v : vector<8xi64>, %shft : i32, %sign : i32) -> vector<8xi32> {
    // CHECK: call <8 x i32> @llvm.aie2.I256.v8.acc64.srs(
    // CHECK-SAME: <8 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I256.v8.acc64.srs"(%v, %shft, %sign) :
                                        (vector<8xi64>, i32, i32) -> vector<8xi32>
    llvm.return %0 : vector<8xi32>
}

// CHECK-LABEL: define <16 x i32> @srs_512b_v16_acc64
llvm.func @srs_512b_v16_acc64(%v : vector<16xi64>, %shft : i32, %sign : i32) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.I512.v16.acc64.srs(
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I512.v16.acc64.srs"(%v, %shft, %sign) : 
                                        (vector<16xi64>, i32, i32) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// CHECK-LABEL: define <32 x i16> @srs_512b_v32_acc32
llvm.func @srs_512b_v32_acc32(%v : vector<16xi64>, %shft : i32, %sign : i32) -> vector<32xi16> {
    // CHECK: call <32 x i16> @llvm.aie2.I512.v32.acc32.srs(
    // CHECK-SAME: <16 x i64> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.I512.v32.acc32.srs"(%v, %shft, %sign) :
                                        (vector<16xi64>, i32, i32) -> vector<32xi16>
    llvm.return %0 : vector<32xi16>
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

// CHECK-LABEL: define <32 x i16> @vbroadcast16_i512
llvm.func @vbroadcast16_i512(%val : i32) -> vector<32xi16> {
    // CHECK: call <32 x i16> @llvm.aie2.vbroadcast16.I512(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vbroadcast16.I512"(%val) : (i32) -> vector<32xi16>
    llvm.return %0 : vector<32xi16>
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

// CHECK-LABEL: define <16 x float> @vbroadcastfloat_i512
llvm.func @vbroadcastfloat_i512(%val : f32) -> vector<16xf32> {
    // CHECK: call <16 x float> @llvm.aie2.vbroadcastfloat.I512(
    // CHECK-SAME: float %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vbroadcastfloat.I512"(%val) : (f32) -> vector<16xf32>
    llvm.return %0 : vector<16xf32>
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

// CHECK-LABEL: define <16 x i32> @ext_i512_i1024
llvm.func @ext_i512_i1024(%v : vector<32xi32>, %idx : i32) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.ext.I512.I1024(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %1 = "xllvm.intr.aie2.ext.I512.I1024"(%v, %idx) : 
                                        (vector<32xi32>, i32) -> vector<16xi32>
    llvm.return %1 : vector<16xi32>
}

// CHECK-LABEL: define <8 x i32> @ext_i256_i1024
llvm.func @ext_i256_i1024(%v : vector<32xi32>, %idx : i32) -> vector<8xi32> {
    // CHECK: call <8 x i32> @llvm.aie2.ext.I256.I1024(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %1 = "xllvm.intr.aie2.ext.I256.I1024"(%v, %idx) : 
                                        (vector<32xi32>, i32) -> vector<8xi32>
    llvm.return %1 : vector<8xi32>
}

// CHECK-LABEL: define <4 x i32> @ext_i128_i512
llvm.func @ext_i128_i512(%v : vector<16xi32>) -> vector<4xi32> {
    // CHECK: call <4 x i32> @llvm.aie2.extract.I128.I512(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}})
    %1 = "xllvm.intr.aie2.extract.I128.I512"(%v) : 
                                        (vector<16xi32>) -> vector<4xi32>
    llvm.return %1 : vector<4xi32>
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

// CHECK-LABEL: define <32 x i32> @concat_i1024_i256
llvm.func @concat_i1024_i256(%a : vector<8xi32>, %b : vector<8xi32>, 
                             %c : vector<8xi32>, %d : vector<8xi32>) -> vector<32xi32> {
    // CHECK: call <32 x i32> @llvm.aie2.concat.I1024.I256(
    // CHECK-SAME: <8 x i32> %{{[0-9]+}}, <8 x i32> %{{[0-9]+}},
    // CHECK-SAME: <8 x i32> %{{[0-9]+}}, <8 x i32> %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.concat.I1024.I256"(%a, %b, %c, %d) : 
            (vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<32xi32>
    llvm.return %0 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @concat_i1024_i512
llvm.func @concat_i1024_i512(%a : vector<16xi32>, %b : vector<16xi32>) -> vector<32xi32> {
    // CHECK: call <32 x i32> @llvm.aie2.concat.I1024.I512(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.concat.I1024.I512"(%a, %b) : 
                                        (vector<16xi32>, vector<16xi32>) -> vector<32xi32>
    llvm.return %0 : vector<32xi32>
}

// ----- SHUFFLE ----- 

// CHECK-LABEL: define <16 x i32> @shuffle_i512
llvm.func @shuffle_i512(%a : vector<16xi32>, %b : vector<16xi32>, %mode : i32) -> vector<16xi32> {
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

// ----- UPD ----- 

// CHECK-LABEL: define <32 x bfloat> @upd_bf512_bf256
llvm.func @upd_bf512_bf256(%a : vector<32xbf16>, %b : vector<16xbf16>, %idx : i32) -> vector<32xbf16> {
    // CHECK: call <32 x bfloat> @llvm.aie2.upd.bf512.bf256(
    // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <16 x bfloat> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.upd.bf512.bf256"(%a, %b, %idx) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
    llvm.return %0 : vector<32xbf16>
}

// ----- SHIFT ----- 

// CHECK-LABEL: define <16 x i32> @vshift_i512_i512
llvm.func @vshift_i512_i512(%a : vector<16xi32>, %b : vector<16xi32>, %step : i32, %shift : i32) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie2.vshift.I512.I512(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vshift.I512.I512"(%a, %b, %step, %shift) : (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// CHECK-LABEL: define <32 x bfloat> @vshift_bf512_bf512
llvm.func @vshift_bf512_bf512(%a : vector<32xbf16>, %b : vector<32xbf16>, %step : i32, %shift : i32) -> vector<32xbf16> {
    // CHECK: call <32 x bfloat> @llvm.aie2.vshift.bf512.bf512(
    // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vshift.bf512.bf512"(%a, %b, %step, %shift) : (vector<32xbf16>, vector<32xbf16>, i32, i32) -> vector<32xbf16>
    llvm.return %0 : vector<32xbf16>
}

// ----- EXTRACT ELEMENT ----- 

// CHECK-LABEL: define i32 @vextract_elem8_i512
llvm.func @vextract_elem8_i512(%a : vector<64xi8>, %idx : i32, %sign : i32) -> i32 {
    // CHECK: call i32 @llvm.aie2.vextract.elem8.I512(
    // CHECK-SAME: <64 x i8> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vextract.elem8.I512"(%a, %idx, %sign) : (vector<64xi8>, i32, i32) -> i32
    llvm.return %0 : i32
}

// CHECK-LABEL: define i32 @vextract_elem16_i512
llvm.func @vextract_elem16_i512(%a : vector<32xi16>, %idx : i32, %sign : i32) -> i32 {
    // CHECK: call i32 @llvm.aie2.vextract.elem16.I512(
    // CHECK-SAME: <32 x i16> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vextract.elem16.I512"(%a, %idx, %sign) : (vector<32xi16>, i32, i32) -> i32
    llvm.return %0 : i32
}

// CHECK-LABEL: define i32 @vextract_elem32_i512
llvm.func @vextract_elem32_i512(%a : vector<16xi32>, %idx : i32, %sign : i32) -> i32 {
    // CHECK: call i32 @llvm.aie2.vextract.elem32.I512(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.vextract.elem32.I512"(%a, %idx, %sign) : (vector<16xi32>, i32, i32) -> i32
    llvm.return %0 : i32
}

// ----- UPS ----- 

// CHECK-LABEL: define <8 x i64> @acc32_v16_i256_ups
llvm.func @acc32_v16_i256_ups(%v : vector<16xi16>, %shift : i32, %sign : i32) -> vector<8xi64> {
    // CHECK: call <8 x i64> @llvm.aie2.acc32.v16.I256.ups(
    // CHECK-SAME: <16 x i16> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.acc32.v16.I256.ups"(%v, %shift, %sign) :
                                        (vector<16xi16>, i32, i32) -> vector<8xi64>
    llvm.return %0 : vector<8xi64>
}

// CHECK-LABEL: define <16 x i64> @acc32_v32_i256_ups
llvm.func @acc32_v32_i256_ups(%v : vector<32xi8>, %shift : i32, %sign : i32) -> vector<16xi64> {
    // CHECK: call <16 x i64> @llvm.aie2.acc32.v32.I256.ups(
    // CHECK-SAME: <32 x i8> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.acc32.v32.I256.ups"(%v, %shift, %sign) :
                                        (vector<32xi8>, i32, i32) -> vector<16xi64>
    llvm.return %0 : vector<16xi64>
}

// CHECK-LABEL: define <16 x i64> @acc32_v32_i512_ups
llvm.func @acc32_v32_i512_ups(%v : vector<32xi16>, %shift : i32, %sign : i32) -> vector<16xi64> {
    // CHECK: call <16 x i64> @llvm.aie2.acc32.v32.I512.ups(
    // CHECK-SAME: <32 x i16> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.acc32.v32.I512.ups"(%v, %shift, %sign) :
                                        (vector<32xi16>, i32, i32) -> vector<16xi64>
    llvm.return %0 : vector<16xi64>
}

// CHECK-LABEL: define <16 x i64> @acc64_v16_i256_ups
llvm.func @acc64_v16_i256_ups(%v : vector<16xi16>, %shift : i32, %sign : i32) -> vector<16xi64> {
    // CHECK: call <16 x i64> @llvm.aie2.acc64.v16.I256.ups(
    // CHECK-SAME: <16 x i16> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.acc64.v16.I256.ups"(%v, %shift, %sign) :
                                        (vector<16xi16>, i32, i32) -> vector<16xi64>
    llvm.return %0 : vector<16xi64>
}

// CHECK-LABEL: define <16 x i64> @acc64_v16_i512_ups
llvm.func @acc64_v16_i512_ups(%v : vector<16xi32>, %shift : i32, %sign : i32) -> vector<16xi64> {
    // CHECK: call <16 x i64> @llvm.aie2.acc64.v16.I512.ups(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.acc64.v16.I512.ups"(%v, %shift, %sign) :
                                        (vector<16xi32>, i32, i32) -> vector<16xi64>
    llvm.return %0 : vector<16xi64>
}

// CHECK-LABEL: define <8 x i64> @acc64_v8_i256_ups
llvm.func @acc64_v8_i256_ups(%v : vector<8xi32>, %shift : i32, %sign : i32) -> vector<8xi64> {
    // CHECK: call <8 x i64> @llvm.aie2.acc64.v8.I256.ups(
    // CHECK-SAME: <8 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.acc64.v8.I256.ups"(%v, %shift, %sign) :
                                        (vector<8xi32>, i32, i32) -> vector<8xi64>
    llvm.return %0 : vector<8xi64>
}

// CHECK-LABEL: define <8 x i64> @accfloat_v16_256b_ups
llvm.func @accfloat_v16_256b_ups(%v : vector<16xbf16>) -> vector<8xi64> {
    // CHECK: call <8 x i64> @llvm.aie2.v16bf16.to.v16accfloat(
    // CHECK-SAME: <16 x bfloat> %{{[0-9]+}})
    %0 = "xllvm.intr.aie2.v16bf16.to.v16accfloat"(%v) : (vector<16xbf16>) -> vector<8xi64>
    llvm.return %0 : vector<8xi64>
}
