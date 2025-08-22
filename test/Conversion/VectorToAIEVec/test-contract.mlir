// RUN: aie-opt %s -convert-vector-to-aievec=aie-target=aie2 | FileCheck %s
// RUN: aie-opt %s -convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" | FileCheck %s --check-prefix=CHECK-LLVM

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @contractbf16bf16f32(
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x8xbf16>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]: vector<8x4xbf16>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xf32>) -> vector<4x4xf32> {
// CHECK:        %[[ACC:.*]] = aievec.cast %[[C]] {isResAcc = true} : vector<4x4xf32>, vector<4x4xf32>
// CHECK:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[ACC]] :
// CHECK-SAME:   vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK:        %[[R:.*]] = aievec.cast %[[MM]] {isResAcc = false} : vector<4x4xf32>, vector<4x4xf32>
// CHECK:        return %[[R]] : vector<4x4xf32>

// CHECK-LLVM-LABEL: func.func @contractbf16bf16f32(
// CHECK-LLVM-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x8xbf16>,
// CHECK-LLVM-SAME: %[[B:[a-zA-Z0-9]+]]: vector<8x4xbf16>,
// CHECK-LLVM-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xf32>) -> vector<4x4xf32> {
// CHECK-LLVM:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-LLVM-SAME:   vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK-LLVM:        return %[[MM]] : vector<4x4xf32>
func.func @contractbf16bf16f32(%A : vector<4x8xbf16>,
                               %B : vector<8x4xbf16>,
                               %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [#map1, #map2, #map3],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %A, %B, %C :
                        vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func.func @contracti16i8i32(
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x4xi16>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]: vector<4x8xi8>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x8xi32>) -> vector<4x8xi32> {
// CHECK:        %[[ACC:.*]] = aievec.cast %[[C]] {isResAcc = true} : vector<4x8xi32>, vector<4x8xi32>
// CHECK:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[ACC]] :
// CHECK-SAME:   vector<4x4xi16>, vector<4x8xi8> into vector<4x8xi32>
// CHECK:        %[[R:.*]] = aievec.cast %[[MM]] {isResAcc = false} : vector<4x8xi32>, vector<4x8xi32>
// CHECK:        return %[[R]] : vector<4x8xi32>

// CHECK-LLVM-LABEL: func.func @contracti16i8i32(
// CHECK-LLVM-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x4xi16>,
// CHECK-LLVM-SAME: %[[B:[a-zA-Z0-9]+]]: vector<4x8xi8>,
// CHECK-LLVM-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x8xi32>) -> vector<4x8xi32> {
// CHECK-LLVM:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-LLVM-SAME:   vector<4x4xi16>, vector<4x8xi8> into vector<4x8xi32>
// CHECK-LLVM:        return %[[MM]] : vector<4x8xi32>
func.func @contracti16i8i32(%A : vector<4x4xi16>,
                            %B : vector<4x8xi8>,
                            %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = arith.extsi %B : vector<4x8xi8> to vector<4x8xi16>
  %1 = vector.contract {indexing_maps = [#map1, #map2, #map3],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %A, %0, %C :
                        vector<4x4xi16>, vector<4x8xi16> into vector<4x8xi32>
  return %1 : vector<4x8xi32>
}

// CHECK-LABEL: func.func @contracti32i16i64(
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x2xi32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]: vector<2x4xi16>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xi64>) -> vector<4x4xi64> {
// CHECK:        %[[ACC:.*]] = aievec.cast %[[C]] {isResAcc = true} : vector<4x4xi64>, vector<4x4xi64>
// CHECK:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[ACC]] :
// CHECK-SAME:   vector<4x2xi32>, vector<2x4xi16> into vector<4x4xi64>
// CHECK:        %[[R:.*]] = aievec.cast %[[MM]] {isResAcc = false} : vector<4x4xi64>, vector<4x4xi64>
// CHECK:        return %[[R]] : vector<4x4xi64>

// CHECK-LLVM-LABEL: func.func @contracti32i16i64(
// CHECK-LLVM-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x2xi32>,
// CHECK-LLVM-SAME: %[[B:[a-zA-Z0-9]+]]: vector<2x4xi16>,
// CHECK-LLVM-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xi64>) -> vector<4x4xi64> {
// CHECK-LLVM:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-LLVM-SAME:   vector<4x2xi32>, vector<2x4xi16> into vector<4x4xi64>
// CHECK-LLVM:        return %[[MM]] : vector<4x4xi64>
func.func @contracti32i16i64(%A : vector<4x2xi32>,
                             %B : vector<2x4xi16>,
                             %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = arith.extsi %B : vector<2x4xi16> to vector<2x4xi32>
  %1 = vector.contract {indexing_maps = [#map1, #map2, #map3],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %A, %0, %C :
                        vector<4x2xi32>, vector<2x4xi32> into vector<4x4xi64>
  return %1 : vector<4x4xi64>
}

// CHECK-LABEL: func.func @contractf32f32f32(
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x8xbf16>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]: vector<8x4xbf16>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xf32>) -> vector<4x4xf32> {
// CHECK:        %[[ACC:.*]] = aievec.cast %[[C]] {isResAcc = true} : vector<4x4xf32>, vector<4x4xf32>
// CHECK:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[ACC]] :
// CHECK-SAME:   vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK:        %[[R:.*]] = aievec.cast %[[MM]] {isResAcc = false} : vector<4x4xf32>, vector<4x4xf32>
// CHECK:        return %[[R]] : vector<4x4xf32>

// CHECK-LLVM-LABEL: func.func @contractf32f32f32(
// CHECK-LLVM-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x8xbf16>,
// CHECK-LLVM-SAME: %[[B:[a-zA-Z0-9]+]]: vector<8x4xbf16>,
// CHECK-LLVM-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xf32>) -> vector<4x4xf32> {
// CHECK-LLVM:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-LLVM-SAME:   vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK-LLVM:        return %[[MM]] : vector<4x4xf32>
func.func @contractf32f32f32(%A : vector<4x8xbf16>,
                             %B : vector<8x4xbf16>,
                             %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = arith.extf %A : vector<4x8xbf16> to vector<4x8xf32>
  %1 = arith.extf %B : vector<8x4xbf16> to vector<8x4xf32>
  %2 = vector.contract {indexing_maps = [#map1, #map2, #map3],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %C :
                        vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  return %2 : vector<4x4xf32>
}

// CHECK-LABEL: func.func @contracti32i32i32(
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x4xi16>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]: vector<4x8xi8>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x8xi32>) -> vector<4x8xi32> {
// CHECK:        %[[ACC:.*]] = aievec.cast %[[C]] {isResAcc = true} : vector<4x8xi32>, vector<4x8xi32>
// CHECK:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[ACC]] :
// CHECK-SAME:   vector<4x4xi16>, vector<4x8xi8> into vector<4x8xi32>
// CHECK:        %[[R:.*]] = aievec.cast %[[MM]] {isResAcc = false} : vector<4x8xi32>, vector<4x8xi32>
// CHECK:        return %[[R]] : vector<4x8xi32>

// CHECK-LLVM-LABEL: func.func @contracti32i32i32(
// CHECK-LLVM-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x4xi16>,
// CHECK-LLVM-SAME: %[[B:[a-zA-Z0-9]+]]: vector<4x8xi8>,
// CHECK-LLVM-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x8xi32>) -> vector<4x8xi32> {
// CHECK-LLVM:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-LLVM-SAME:   vector<4x4xi16>, vector<4x8xi8> into vector<4x8xi32>
// CHECK-LLVM:        return %[[MM]] : vector<4x8xi32>
func.func @contracti32i32i32(%A : vector<4x4xi16>,
                             %B : vector<4x8xi8>,
                             %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = arith.extsi %A : vector<4x4xi16> to vector<4x4xi32>
  %1 = arith.extsi %B : vector<4x8xi8> to vector<4x8xi32>
  %2 = vector.contract {indexing_maps = [#map1, #map2, #map3],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %C :
                        vector<4x4xi32>, vector<4x8xi32> into vector<4x8xi32>
  return %2 : vector<4x8xi32>
}

// CHECK-LABEL: func.func @contracti64i64i64(
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x2xi32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]: vector<2x4xi16>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xi64>) -> vector<4x4xi64> {
// CHECK:        %[[ACC:.*]] = aievec.cast %[[C]] {isResAcc = true} : vector<4x4xi64>, vector<4x4xi64>
// CHECK:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[ACC]] :
// CHECK-SAME:   vector<4x2xi32>, vector<2x4xi16> into vector<4x4xi64>
// CHECK:        %[[R:.*]] = aievec.cast %[[MM]] {isResAcc = false} : vector<4x4xi64>, vector<4x4xi64>
// CHECK:        return %[[R]] : vector<4x4xi64>

// CHECK-LLVM-LABEL: func.func @contracti64i64i64(
// CHECK-LLVM-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x2xi32>,
// CHECK-LLVM-SAME: %[[B:[a-zA-Z0-9]+]]: vector<2x4xi16>,
// CHECK-LLVM-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xi64>) -> vector<4x4xi64> {
// CHECK-LLVM:        %[[MM:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-LLVM-SAME:   vector<4x2xi32>, vector<2x4xi16> into vector<4x4xi64>
// CHECK-LLVM:        return %[[MM]] : vector<4x4xi64>
func.func @contracti64i64i64(%A : vector<4x2xi32>,
                             %B : vector<2x4xi16>,
                             %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = arith.extui %A : vector<4x2xi32> to vector<4x2xi64>
  %1 = arith.extui %B : vector<2x4xi16> to vector<2x4xi64>
  %2 = vector.contract {indexing_maps = [#map1, #map2, #map3],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %C :
                        vector<4x2xi64>, vector<2x4xi64> into vector<4x4xi64>
  return %2 : vector<4x4xi64>
}

