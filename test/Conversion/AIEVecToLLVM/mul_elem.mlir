// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie2-fp32-emulation-strategy=accuracy-fast" | FileCheck --check-prefix=FP32FAST %s
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie2-fp32-emulation-strategy=accuracy-low" | FileCheck --check-prefix=FP32LOW %s

func.func @i16_i16_i32_mul_elem(%arg0 : vector<32xi16>, %arg1 : vector<32xi16>) -> vector<32xi32> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<32xi16>, vector<32xi16>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @i16_i16_i32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xi16>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(824 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi16> to vector<64xi8>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[MULCONF:.*]] = "xllvm.intr.aie2.I512.I512.acc32.mul.conf"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[CST]]) : 
// CHECK-SAME: (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[MULCONF]] : vector<16xi64> to vector<32xi32>
// CHECK-NEXT: return %[[RES]] : vector<32xi32>

// -----

func.func @i8_i8_i32_mul_elem(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>) -> vector<32xi32> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<32xi8>, vector<32xi8>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @i8_i8_i32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi8>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xi8>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(808 : i32) : i32
// CHECK: %[[C0I32:.*]] = llvm.mlir.constant(0 : i32) : i32 
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi8> to vector<8xi32>
// CHECK: %[[IFA0:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BITCAST0]],
// CHECK-SAME:            %[[C0I32]]) : (vector<8xi32>, i32) -> vector<16xi32>
// CHECK: %[[BITCAST1:.*]] = llvm.bitcast %[[IFA0]] : vector<16xi32> to vector<64xi8>
// CHECK: %[[C1I32:.*]] = llvm.mlir.constant(0 : i32) : i32 
// CHECK: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi8> to vector<8xi32> 
// CHECK: %[[IFA1:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BITCAST2]],
// CHECK-SAME:            %[[C1I32]]) : (vector<8xi32>, i32) -> vector<16xi32>
// CHECK: %[[BITCAST3:.*]] = llvm.bitcast %[[IFA1]] : vector<16xi32> to vector<16xi32>
// CHECK: %[[MULCONF:.*]] = "xllvm.intr.aie2.I512.I512.acc32.mul.conf"(
// CHECK-SAME: %[[BITCAST1]], %[[BITCAST3]], %[[CST]]) : 
// CHECK-SAME: (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
// CHECK: %[[RES:.*]] = llvm.bitcast %[[MULCONF]] : vector<16xi64> to vector<32xi32> 
// CHECK-NEXT: return %[[RES]] : vector<32xi32> 

// -----

func.func @i32_i32_i32_mul_elem(%arg0 : vector<16xi32>, %arg1 : vector<16xi32>) -> vector<16xi64> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<16xi32>, vector<16xi32>, vector<16xi64>
  return %0 : vector<16xi64>
}

// CHECK-LABEL: @i32_i32_i32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xi32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VBROADCAST:.*]] = "xllvm.intr.aie2.vbroadcast32.I512"(%[[CST0]]) : (i32) -> vector<16xi32>
// CHECK-NEXT: %[[UNDEF:.*]]  = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[SHUFF0:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG0]], %[[VBROADCAST]], %[[CST1]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST2:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[SHUFF1:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG0]], %[[VBROADCAST]], %[[CST2]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST3:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[SHUFF2:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG1]], %[[UNDEF]], %[[CST3]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST4:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[SHUFF3:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG1]], %[[UNDEF]], %[[CST4]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST5:.*]] = llvm.mlir.constant(858 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[SHUFF1]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC0:.*]] = "xllvm.intr.aie2.I512.I512.acc64.mul.conf"(%[[BITCAST0]], %[[SHUFF3]], %[[CST5]]) : (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[CST6:.*]] = llvm.mlir.constant(1626 : i32) : i32
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SHUFF1]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC1:.*]] = "xllvm.intr.aie2.I512.I512.ACC1024.acc64.mac.conf"(%[[BITCAST1]], %[[SHUFF2]], %[[ACC0]], %[[CST6]]) : (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[CST7:.*]] = llvm.mlir.constant(346 : i32) : i32
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[SHUFF0]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC2:.*]] = "xllvm.intr.aie2.I512.I512.ACC1024.acc64.mac.conf"(%[[BITCAST2]], %[[SHUFF3]], %[[ACC1]], %[[CST7]]) : (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[CST8:.*]] = llvm.mlir.constant(1114 : i32) : i32
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[SHUFF0]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC3:.*]] = "xllvm.intr.aie2.I512.I512.ACC1024.acc64.mac.conf"(%[[BITCAST3]], %[[SHUFF2]], %[[ACC2]], %[[CST8]]) : (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[ACC3]] : vector<16xi64> to vector<16xi64>
// CHECK-NEXT: return %[[RES]] : vector<16xi64>

// -----

func.func @bf16_bf16_f32_mul_elem(%arg0 : vector<16xbf16>, %arg1 : vector<16xbf16>) -> vector<16xf32> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @bf16_bf16_f32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xbf16>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK: %[[C0I32:.*]] = llvm.mlir.constant(0 : i32) : i32 
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xbf16> to vector<8xi32>
// CHECK: %[[IFA0:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BITCAST0]],
// CHECK-SAME:            %[[C0I32]]) : (vector<8xi32>, i32) -> vector<16xi32>
// CHECK: %[[BITCAST1:.*]] = llvm.bitcast %[[IFA0]] : vector<16xi32> to vector<32xbf16>
// CHECK: %[[C1I32:.*]] = llvm.mlir.constant(0 : i32) : i32 
// CHECK: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG1]] : vector<16xbf16> to vector<8xi32> 
// CHECK: %[[IFA1:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BITCAST2]],
// CHECK-SAME:            %[[C1I32]]) : (vector<8xi32>, i32) -> vector<16xi32>
// CHECK: %[[BITCAST3:.*]] = llvm.bitcast %[[IFA1]] : vector<16xi32> to vector<32xbf16>
// CHECK-NEXT: %[[MULCONF:.*]] = "xllvm.intr.aie2.bf.mul16.conf"(
// CHECK-SAME: %[[BITCAST1]], %[[BITCAST3]], %[[CST]]) : 
// CHECK-SAME: (vector<32xbf16>, vector<32xbf16>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[MULCONF]] : vector<8xi64> to vector<16xf32>
// CHECK-NEXT: return %[[RES]] : vector<16xf32>
// -----

func.func @f32_f32_f32_mul_elem(%arg0 : vector<16xf32>, %arg1 : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<16xf32>, vector<16xf32>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @f32_f32_f32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xf32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
// CHECK-NEXT: %[[ZEROS0:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// CHECK-NEXT: %[[ZEROS1:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// CHECK-NEXT: %[[ZEROS2:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// CHECK-NEXT: %[[ZEROS3:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// CHECK-NEXT: %[[ZEROS4:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// CHECK-NEXT: %[[ZEROS5:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1.000000e+00 : bf16) : bf16
// CHECK-NEXT: %[[ONES0:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST1]]) : (bf16) -> vector<32xbf16>
// CHECK-NEXT: %[[CST2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<8xi64>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[BITCAST0]]) : (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[UPD0:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS0]], %[[SRS0]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// CHECK-NEXT: %[[MSC0:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD0]], %[[ONES0]], %[[BITCAST0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC0]]) : (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[UPD1:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS1]], %[[SRS1]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// CHECK-NEXT: %[[MSC1:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD1]], %[[ONES0]], %[[MSC0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[SRS2:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC1]]) : (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[UPD2:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS2]], %[[SRS2]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xf32> to vector<8xi64>
// CHECK-NEXT: %[[SRS3:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[BITCAST1]]) : (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[UPD3:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS3]], %[[SRS3]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// CHECK-NEXT: %[[MSC2:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD3]], %[[ONES0]], %[[BITCAST1]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[SRS4:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC2]]) : (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[UPD4:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS4]], %[[SRS4]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// CHECK-NEXT: %[[MSC3:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD4]], %[[ONES0]], %[[MSC2]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[SRS5:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC3]]) : (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[UPD5:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS5]], %[[SRS5]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// CHECK-NEXT: %[[ACC0:.*]] = "xllvm.intr.aie2.bf.mul16.conf"(%[[UPD2]], %[[UPD5]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC1:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD2]], %[[UPD4]], %[[ACC0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC2:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD1]], %[[UPD5]], %[[ACC1]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC3:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD0]], %[[UPD5]], %[[ACC2]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC4:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD1]], %[[UPD4]], %[[ACC3]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC5:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD3]], %[[UPD2]], %[[ACC4]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC6:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD1]], %[[UPD3]], %[[ACC5]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC7:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD0]], %[[UPD4]], %[[ACC6]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[ACC8:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD0]], %[[UPD3]], %[[ACC7]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[ACC8]] : vector<8xi64> to vector<16xf32>
// CHECK-NEXT: return %[[RES]] : vector<16xf32>

// FP32FAST-LABEL: @f32_f32_f32_mul_elem
// FP32FAST-SAME: %[[ARG0:.*]]: vector<16xf32>,
// FP32FAST-SAME: %[[ARG1:.*]]: vector<16xf32>
// FP32FAST: %[[CST0:.*]] = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
// FP32FAST-NEXT: %[[ZEROS0:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32FAST-NEXT: %[[ZEROS1:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32FAST-NEXT: %[[ZEROS2:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32FAST-NEXT: %[[ZEROS3:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32FAST-NEXT: %[[ZEROS4:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32FAST-NEXT: %[[ZEROS5:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32FAST-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1.000000e+00 : bf16) : bf16
// FP32FAST-NEXT: %[[ONES0:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST1]]) : (bf16) -> vector<32xbf16>
// FP32FAST-NEXT: %[[CST2:.*]] = llvm.mlir.constant(0 : i32) : i32
// FP32FAST-NEXT: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// FP32FAST-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<8xi64>
// FP32FAST-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[BITCAST0]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32FAST-NEXT: %[[UPD0:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS0]], %[[SRS0]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32FAST-NEXT: %[[MSC0:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD0]], %[[ONES0]], %[[BITCAST0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC0]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32FAST-NEXT: %[[UPD1:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS1]], %[[SRS1]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32FAST-NEXT: %[[MSC1:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD1]], %[[ONES0]], %[[MSC0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[SRS2:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC1]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32FAST-NEXT: %[[UPD2:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS2]], %[[SRS2]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32FAST-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xf32> to vector<8xi64>
// FP32FAST-NEXT: %[[SRS3:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[BITCAST1]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32FAST-NEXT: %[[UPD3:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS3]], %[[SRS3]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32FAST-NEXT: %[[MSC2:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD3]], %[[ONES0]], %[[BITCAST1]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[SRS4:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC2]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32FAST-NEXT: %[[UPD4:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS4]], %[[SRS4]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32FAST-NEXT: %[[MSC3:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD4]], %[[ONES0]], %[[MSC2]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[SRS5:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC3]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32FAST-NEXT: %[[UPD5:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS5]], %[[SRS5]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32FAST-NEXT: %[[ACC0:.*]] = "xllvm.intr.aie2.bf.mul16.conf"(%[[UPD0]], %[[UPD5]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[ACC1:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD1]], %[[UPD4]], %[[ACC0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[ACC2:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD3]], %[[UPD2]], %[[ACC1]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[ACC3:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD1]], %[[UPD3]], %[[ACC2]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[ACC4:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD0]], %[[UPD4]], %[[ACC3]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[ACC5:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD0]], %[[UPD3]], %[[ACC4]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32FAST-NEXT: %[[RES:.*]] = llvm.bitcast %[[ACC5]] : vector<8xi64> to vector<16xf32>
// FP32FAST-NEXT: return %[[RES]] : vector<16xf32>

// FP32LOW-LABEL: @f32_f32_f32_mul_elem
// FP32LOW-SAME: %[[ARG0:.*]]: vector<16xf32>,
// FP32LOW-SAME: %[[ARG1:.*]]: vector<16xf32>
// FP32LOW: %[[CST0:.*]] = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
// FP32LOW-NEXT: %[[ZEROS0:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32LOW-NEXT: %[[ZEROS1:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32LOW-NEXT: %[[ZEROS2:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32LOW-NEXT: %[[ZEROS3:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32LOW-NEXT: %[[ZEROS4:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32LOW-NEXT: %[[ZEROS5:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST0]]) : (bf16) -> vector<32xbf16>
// FP32LOW-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1.000000e+00 : bf16) : bf16
// FP32LOW-NEXT: %[[ONES0:.*]] = "xllvm.intr.aie2.vbroadcast16.bf512"(%[[CST1]]) : (bf16) -> vector<32xbf16>
// FP32LOW-NEXT: %[[CST2:.*]] = llvm.mlir.constant(0 : i32) : i32
// FP32LOW-NEXT: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// FP32LOW-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<8xi64>
// FP32LOW-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[BITCAST0]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32LOW-NEXT: %[[UPD0:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS0]], %[[SRS0]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32LOW-NEXT: %[[MSC0:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD0]], %[[ONES0]], %[[BITCAST0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32LOW-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC0]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32LOW-NEXT: %[[UPD1:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS1]], %[[SRS1]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32LOW-NEXT: %[[MSC1:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD1]], %[[ONES0]], %[[MSC0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32LOW-NEXT: %[[SRS2:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC1]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32LOW-NEXT: %[[UPD2:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS2]], %[[SRS2]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32LOW-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xf32> to vector<8xi64>
// FP32LOW-NEXT: %[[SRS3:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[BITCAST1]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32LOW-NEXT: %[[UPD3:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS3]], %[[SRS3]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32LOW-NEXT: %[[MSC2:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD3]], %[[ONES0]], %[[BITCAST1]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32LOW-NEXT: %[[SRS4:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC2]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32LOW-NEXT: %[[UPD4:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS4]], %[[SRS4]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32LOW-NEXT: %[[MSC3:.*]] = "xllvm.intr.aie2.bf.msc16.conf"(%[[UPD4]], %[[ONES0]], %[[MSC2]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32LOW-NEXT: %[[SRS5:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(%[[MSC3]]) : (vector<8xi64>) -> vector<16xbf16>
// FP32LOW-NEXT: %[[UPD5:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[ZEROS5]], %[[SRS5]], %[[CST2]]) : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
// FP32LOW-NEXT: %[[ACC0:.*]] = "xllvm.intr.aie2.bf.mul16.conf"(%[[UPD1]], %[[UPD3]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, i32) -> vector<8xi64>
// FP32LOW-NEXT: %[[ACC1:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD0]], %[[UPD4]], %[[ACC0]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32LOW-NEXT: %[[ACC2:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[UPD0]], %[[UPD3]], %[[ACC1]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, vector<8xi64>, i32) -> vector<8xi64>
// FP32LOW-NEXT: %[[RES:.*]] = llvm.bitcast %[[ACC2]] : vector<8xi64> to vector<16xf32>
// FP32LOW-NEXT: return %[[RES]] : vector<16xf32>
