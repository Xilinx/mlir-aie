// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// CHECK-LABEL:func @reduce_add_i32
// CHECK-SAME: %[[SRC:.*]]: vector<16xi32>
func.func @reduce_add_i32(%arg0: vector<16xi32>) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[ADD1:.*]] = aievec.add_elem %[[SRC]], %[[SHIFT32]] : vector<16xi32>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[ADD1]], %[[ADD1]], %[[C16]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[ADD2:.*]] = aievec.add_elem %[[ADD1]], %[[SHIFT16]] : vector<16xi32>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[ADD2]], %[[ADD2]], %[[C8]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[ADD3:.*]] = aievec.add_elem %[[ADD2]], %[[SHIFT8]] : vector<16xi32>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[ADD3]], %[[ADD3]], %[[C4]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[ADD4:.*]] = aievec.add_elem %[[ADD3]], %[[SHIFT4]] : vector<16xi32>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[ADD4]], %[[C0]] : vector<16xi32>, i32, i32
  %0 = vector.reduction <add>, %arg0 : vector<16xi32> into i32
  // CHECK: return %[[EXTELEM]] : i32
  return %0 : i32
}

// CHECK-LABEL:func @reduce_min_i32
// CHECK-SAME: %[[SRC:.*]]: vector<16xi32>
func.func @reduce_min_i32(%arg0: vector<16xi32>) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MIN1:.*]] = aievec.min %[[SRC]], %[[SHIFT32]] : vector<16xi32>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MIN1]], %[[MIN1]], %[[C16]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MIN2:.*]] = aievec.min %[[MIN1]], %[[SHIFT16]] : vector<16xi32>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MIN2]], %[[MIN2]], %[[C8]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MIN3:.*]] = aievec.min %[[MIN2]], %[[SHIFT8]] : vector<16xi32>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MIN3]], %[[MIN3]], %[[C4]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MIN4:.*]] = aievec.min %[[MIN3]], %[[SHIFT4]] : vector<16xi32>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MIN4]], %[[C0]] : vector<16xi32>, i32, i32
  %0 = vector.reduction <minsi>, %arg0 : vector<16xi32> into i32
  // CHECK: return %[[EXTELEM]] : i32
  return %0 : i32
}

// CHECK-LABEL:func @reduce_max_i32
// CHECK-SAME: %[[SRC:.*]]: vector<16xi32>
func.func @reduce_max_i32(%arg0: vector<16xi32>) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MAX1:.*]] = aievec.max %[[SRC]], %[[SHIFT32]] : vector<16xi32>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MAX1]], %[[MAX1]], %[[C16]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MAX2:.*]] = aievec.max %[[MAX1]], %[[SHIFT16]] : vector<16xi32>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MAX2]], %[[MAX2]], %[[C8]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MAX3:.*]] = aievec.max %[[MAX2]], %[[SHIFT8]] : vector<16xi32>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MAX3]], %[[MAX3]], %[[C4]] {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  // CHECK: %[[MAX4:.*]] = aievec.max %[[MAX3]], %[[SHIFT4]] : vector<16xi32>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MAX4]], %[[C0]] : vector<16xi32>, i32, i32
  %0 = vector.reduction <maxsi>, %arg0 : vector<16xi32> into i32
  // CHECK: return %[[EXTELEM]] : i32
  return %0 : i32
}

// CHECK-LABEL:func @reduce_add_i16
// CHECK-SAME: %[[SRC:.*]]: vector<32xi16>
func.func @reduce_add_i16(%arg0: vector<32xi16>) -> i16 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[ADD1:.*]] = aievec.add_elem %[[SRC]], %[[SHIFT32]] : vector<32xi16>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[ADD1]], %[[ADD1]], %[[C16]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[ADD2:.*]] = aievec.add_elem %[[ADD1]], %[[SHIFT16]] : vector<32xi16>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[ADD2]], %[[ADD2]], %[[C8]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[ADD3:.*]] = aievec.add_elem %[[ADD2]], %[[SHIFT8]] : vector<32xi16>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[ADD3]], %[[ADD3]], %[[C4]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[ADD4:.*]] = aievec.add_elem %[[ADD3]], %[[SHIFT4]] : vector<32xi16>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[ADD4]], %[[ADD4]], %[[C2]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[ADD5:.*]] = aievec.add_elem %[[ADD4]], %[[SHIFT5]] : vector<32xi16>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[ADD5]], %[[C0]] : vector<32xi16>, i32, i16
  %0 = vector.reduction <add>, %arg0 : vector<32xi16> into i16
  // CHECK: return %[[EXTELEM]] : i16
  return %0 : i16
}

// CHECK-LABEL:func @reduce_min_i16
// CHECK-SAME: %[[SRC:.*]]: vector<32xi16>
func.func @reduce_min_i16(%arg0: vector<32xi16>) -> i16 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MIN1:.*]] = aievec.min %[[SRC]], %[[SHIFT32]] : vector<32xi16>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MIN1]], %[[MIN1]], %[[C16]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MIN2:.*]] = aievec.min %[[MIN1]], %[[SHIFT16]] : vector<32xi16>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MIN2]], %[[MIN2]], %[[C8]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MIN3:.*]] = aievec.min %[[MIN2]], %[[SHIFT8]] : vector<32xi16>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MIN3]], %[[MIN3]], %[[C4]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MIN4:.*]] = aievec.min %[[MIN3]], %[[SHIFT4]] : vector<32xi16>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[MIN4]], %[[MIN4]], %[[C2]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MIN5:.*]] = aievec.min %[[MIN4]], %[[SHIFT5]] : vector<32xi16>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MIN5]], %[[C0]] : vector<32xi16>, i32, i16
  %0 = vector.reduction <minsi>, %arg0 : vector<32xi16> into i16
  // CHECK: return %[[EXTELEM]] : i16
  return %0 : i16
}

// CHECK-LABEL:func @reduce_max_i16
// CHECK-SAME: %[[SRC:.*]]: vector<32xi16>
func.func @reduce_max_i16(%arg0: vector<32xi16>) -> i16 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MAX1:.*]] = aievec.max %[[SRC]], %[[SHIFT32]] : vector<32xi16>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MAX1]], %[[MAX1]], %[[C16]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MAX2:.*]] = aievec.max %[[MAX1]], %[[SHIFT16]] : vector<32xi16>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MAX2]], %[[MAX2]], %[[C8]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MAX3:.*]] = aievec.max %[[MAX2]], %[[SHIFT8]] : vector<32xi16>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MAX3]], %[[MAX3]], %[[C4]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MAX4:.*]] = aievec.max %[[MAX3]], %[[SHIFT4]] : vector<32xi16>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[MAX4]], %[[MAX4]], %[[C2]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  // CHECK: %[[MAX5:.*]] = aievec.max %[[MAX4]], %[[SHIFT5]] : vector<32xi16>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MAX5]], %[[C0]] : vector<32xi16>, i32, i16
  %0 = vector.reduction <maxsi>, %arg0 : vector<32xi16> into i16
  // CHECK: return %[[EXTELEM]] : i16
  return %0 : i16
}

// CHECK-LABEL:func @reduce_add_i8
// CHECK-SAME: %[[SRC:.*]]: vector<64xi8>
func.func @reduce_add_i8(%arg0: vector<64xi8>) -> i8 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[ADD1:.*]] = aievec.add_elem %[[SRC]], %[[SHIFT32]] : vector<64xi8>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[ADD1]], %[[ADD1]], %[[C16]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[ADD2:.*]] = aievec.add_elem %[[ADD1]], %[[SHIFT16]] : vector<64xi8>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[ADD2]], %[[ADD2]], %[[C8]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[ADD3:.*]] = aievec.add_elem %[[ADD2]], %[[SHIFT8]] : vector<64xi8>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[ADD3]], %[[ADD3]], %[[C4]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[ADD4:.*]] = aievec.add_elem %[[ADD3]], %[[SHIFT4]] : vector<64xi8>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[ADD4]], %[[ADD4]], %[[C2]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[ADD5:.*]] = aievec.add_elem %[[ADD4]], %[[SHIFT5]] : vector<64xi8>
  // CHECK: %[[SHIFT6:.*]] = aievec.shift %[[ADD5]], %[[ADD5]], %[[C1]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[ADD6:.*]] = aievec.add_elem %[[ADD5]], %[[SHIFT6]] : vector<64xi8>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[ADD6]], %[[C0]] : vector<64xi8>, i32, i8
  %0 = vector.reduction <add>, %arg0 : vector<64xi8> into i8
  // CHECK: return %[[EXTELEM]] : i8
  return %0 : i8
}

// CHECK-LABEL:func @reduce_min_i8
// CHECK-SAME: %[[SRC:.*]]: vector<64xi8>
func.func @reduce_min_i8(%arg0: vector<64xi8>) -> i8 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MIN1:.*]] = aievec.min %[[SRC]], %[[SHIFT32]] : vector<64xi8>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MIN1]], %[[MIN1]], %[[C16]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MIN2:.*]] = aievec.min %[[MIN1]], %[[SHIFT16]] : vector<64xi8>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MIN2]], %[[MIN2]], %[[C8]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MIN3:.*]] = aievec.min %[[MIN2]], %[[SHIFT8]] : vector<64xi8>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MIN3]], %[[MIN3]], %[[C4]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MIN4:.*]] = aievec.min %[[MIN3]], %[[SHIFT4]] : vector<64xi8>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[MIN4]], %[[MIN4]], %[[C2]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MIN5:.*]] = aievec.min %[[MIN4]], %[[SHIFT5]] : vector<64xi8>
  // CHECK: %[[SHIFT6:.*]] = aievec.shift %[[MIN5]], %[[MIN5]], %[[C1]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MIN6:.*]] = aievec.min %[[MIN5]], %[[SHIFT6]] : vector<64xi8>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MIN6]], %[[C0]] : vector<64xi8>, i32, i8
  %0 = vector.reduction <minsi>, %arg0 : vector<64xi8> into i8
  // CHECK: return %[[EXTELEM]] : i8
  return %0 : i8
}

// CHECK-LABEL:func @reduce_max_i8
// CHECK-SAME: %[[SRC:.*]]: vector<64xi8>
func.func @reduce_max_i8(%arg0: vector<64xi8>) -> i8 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MAX1:.*]] = aievec.max %[[SRC]], %[[SHIFT32]] : vector<64xi8>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MAX1]], %[[MAX1]], %[[C16]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MAX2:.*]] = aievec.max %[[MAX1]], %[[SHIFT16]] : vector<64xi8>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MAX2]], %[[MAX2]], %[[C8]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MAX3:.*]] = aievec.max %[[MAX2]], %[[SHIFT8]] : vector<64xi8>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MAX3]], %[[MAX3]], %[[C4]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MAX4:.*]] = aievec.max %[[MAX3]], %[[SHIFT4]] : vector<64xi8>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[MAX4]], %[[MAX4]], %[[C2]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MAX5:.*]] = aievec.max %[[MAX4]], %[[SHIFT5]] : vector<64xi8>
  // CHECK: %[[SHIFT6:.*]] = aievec.shift %[[MAX5]], %[[MAX5]], %[[C1]] {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  // CHECK: %[[MAX6:.*]] = aievec.max %[[MAX5]], %[[SHIFT6]] : vector<64xi8>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MAX6]], %[[C0]] : vector<64xi8>, i32, i8
  %0 = vector.reduction <maxsi>, %arg0 : vector<64xi8> into i8
  // CHECK: return %[[EXTELEM]] : i8
  return %0 : i8
}

// CHECK-LABEL:func @reduce_add_f32
// CHECK-SAME: %[[SRC:.*]]: vector<16xf32>
func.func @reduce_add_f32(%arg0: vector<16xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[CASTL1:.*]] = aievec.cast %[[SRC]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[CASTR1:.*]] = aievec.cast %[[SHIFT32]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[ADD1:.*]] = aievec.add_elem %[[CASTL1]], %[[CASTR1]] : vector<16xf32>
  // CHECK: %[[CAST1:.*]] = aievec.cast %[[ADD1]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[CAST1]], %[[CAST1]], %[[C16]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[CASTR2:.*]] = aievec.cast %[[SHIFT16]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[ADD2:.*]] = aievec.add_elem %[[ADD1]], %[[CASTR2]] : vector<16xf32>
  // CHECK: %[[CAST2:.*]] = aievec.cast %[[ADD2]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[CAST2]], %[[CAST2]], %[[C8]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[CASTR3:.*]] = aievec.cast %[[SHIFT8]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[ADD3:.*]] = aievec.add_elem %[[ADD2]], %[[CASTR3]] : vector<16xf32>
  // CHECK: %[[CAST3:.*]] = aievec.cast %[[ADD3]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[CAST3]], %[[CAST3]], %[[C4]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[CASTR4:.*]] = aievec.cast %[[SHIFT4]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[ADD4:.*]] = aievec.add_elem %[[ADD3]], %[[CASTR4]] : vector<16xf32>
  // CHECK: %[[CAST4:.*]] = aievec.cast %[[ADD4]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[CAST4]], %[[C0]] : vector<16xf32>, i32, f32
  %0 = vector.reduction <add>, %arg0 : vector<16xf32> into f32
  // CHECK: return %[[EXTELEM]] : f32
  return %0 : f32
}

// CHECK-LABEL:func @reduce_min_f32
// CHECK-SAME: %[[SRC:.*]]: vector<16xf32>
func.func @reduce_min_f32(%arg0: vector<16xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MIN1:.*]] = aievec.min %[[SRC]], %[[SHIFT32]] : vector<16xf32>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MIN1]], %[[MIN1]], %[[C16]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MIN2:.*]] = aievec.min %[[MIN1]], %[[SHIFT16]] : vector<16xf32>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MIN2]], %[[MIN2]], %[[C8]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MIN3:.*]] = aievec.min %[[MIN2]], %[[SHIFT8]] : vector<16xf32>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MIN3]], %[[MIN3]], %[[C4]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MIN4:.*]] = aievec.min %[[MIN3]], %[[SHIFT4]] : vector<16xf32>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MIN4]], %[[C0]] : vector<16xf32>, i32, f32 
  %0 = vector.reduction <minimumf>, %arg0 : vector<16xf32> into f32
  // CHECK: return %[[EXTELEM]] : f32
  return %0 : f32
}

// CHECK-LABEL:func @reduce_max_f32
// CHECK-SAME: %[[SRC:.*]]: vector<16xf32>
func.func @reduce_max_f32(%arg0: vector<16xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MAX1:.*]] = aievec.max %[[SRC]], %[[SHIFT32]] : vector<16xf32>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MAX1]], %[[MAX1]], %[[C16]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MAX2:.*]] = aievec.max %[[MAX1]], %[[SHIFT16]] : vector<16xf32>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MAX2]], %[[MAX2]], %[[C8]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MAX3:.*]] = aievec.max %[[MAX2]], %[[SHIFT8]] : vector<16xf32>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MAX3]], %[[MAX3]], %[[C4]] {isAcc = false} : vector<16xf32>, vector<16xf32>, i32, vector<16xf32>
  // CHECK: %[[MAX4:.*]] = aievec.max %[[MAX3]], %[[SHIFT4]] : vector<16xf32>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MAX4]], %[[C0]] : vector<16xf32>, i32, f32
  %0 = vector.reduction <maximumf>, %arg0 : vector<16xf32> into f32
  // CHECK: return %[[EXTELEM]] : f32
  return %0 : f32
}

// CHECK-LABEL:func @reduce_min_bf16
// CHECK-SAME: %[[SRC:.*]]: vector<32xbf16>
func.func @reduce_min_bf16(%arg0: vector<32xbf16>) -> bf16 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MIN1:.*]] = aievec.min %[[SRC]], %[[SHIFT32]] : vector<32xbf16>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MIN1]], %[[MIN1]], %[[C16]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MIN2:.*]] = aievec.min %[[MIN1]], %[[SHIFT16]] : vector<32xbf16>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MIN2]], %[[MIN2]], %[[C8]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MIN3:.*]] = aievec.min %[[MIN2]], %[[SHIFT8]] : vector<32xbf16>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MIN3]], %[[MIN3]], %[[C4]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MIN4:.*]] = aievec.min %[[MIN3]], %[[SHIFT4]] : vector<32xbf16>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[MIN4]], %[[MIN4]], %[[C2]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MIN5:.*]] = aievec.min %[[MIN4]], %[[SHIFT5]] : vector<32xbf16>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MAX5]], %[[C0]] : vector<32xbf16>, i32, bf16
  %0 = vector.reduction <minimumf>, %arg0 : vector<32xbf16> into bf16
  // CHECK: return %[[EXTELEM]] : bf16
  return %0 : bf16
}

// CHECK-LABEL:func @reduce_max_bf16
// CHECK-SAME: %[[SRC:.*]]: vector<32xbf16>
func.func @reduce_max_bf16(%arg0: vector<32xbf16>) -> bf16 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK: %[[C32:.*]] = arith.constant 32 : i32
  // CHECK: %[[SHIFT32:.*]] = aievec.shift %[[SRC]], %[[SRC]], %[[C32]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MAX1:.*]] = aievec.max %[[SRC]], %[[SHIFT32]] : vector<32xbf16>
  // CHECK: %[[SHIFT16:.*]] = aievec.shift %[[MAX1]], %[[MAX1]], %[[C16]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MAX2:.*]] = aievec.max %[[MAX1]], %[[SHIFT16]] : vector<32xbf16>
  // CHECK: %[[SHIFT8:.*]] = aievec.shift %[[MAX2]], %[[MAX2]], %[[C8]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MAX3:.*]] = aievec.max %[[MAX2]], %[[SHIFT8]] : vector<32xbf16>
  // CHECK: %[[SHIFT4:.*]] = aievec.shift %[[MAX3]], %[[MAX3]], %[[C4]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MAX4:.*]] = aievec.max %[[MAX3]], %[[SHIFT4]] : vector<32xbf16>
  // CHECK: %[[SHIFT5:.*]] = aievec.shift %[[MAX4]], %[[MAX4]], %[[C2]] {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  // CHECK: %[[MAX5:.*]] = aievec.max %[[MAX4]], %[[SHIFT5]] : vector<32xbf16>
  // CHECK: %[[EXTELEM:.*]] = aievec.ext_elem %[[MAX5]], %[[C0]] : vector<32xbf16>, i32, bf16
  %0 = vector.reduction <maximumf>, %arg0 : vector<32xbf16> into bf16
  // CHECK: return %[[EXTELEM]] : bf16
  return %0 : bf16
}
