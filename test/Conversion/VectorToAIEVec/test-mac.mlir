// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s

// CHECK-LABEL: func.func @muladd2mac_i32(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @muladd2mac_i32(%a : vector<8xi32>,
                          %b : vector<8xi32>,
                          %c : vector<8xi32>) -> vector<8xi32> {
    // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[AA:.*]] = aievec.concat %[[A]], %[[A]] : vector<8xi32>, vector<16xi32>
    // CHECK: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK: %[[MAC:.*]] = aievec_aie1.mac %[[AA]], %[[B]], %[[ACC]] : vector<16xi32>, vector<8xi32>, vector<8xi80>
    // CHECK: %[[RES:.*]] = aievec.srs %[[MAC]], %[[C0]] : vector<8xi80>, i32, vector<8xi32>
    %0 = arith.muli %a, %b : vector<8xi32>
    %1 = arith.addi %0, %c : vector<8xi32>
    // CHECK: return %[[RES]] : vector<8xi32>
    return %1 : vector<8xi32>
}

// CHECK-LABEL: func.func @muladd2mac_inv(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @muladd2mac_inv(%a : vector<8xi32>,
                          %b : vector<8xi32>,
                          %c : vector<8xi32>) -> vector<8xi32> {
    // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[AA:.*]] = aievec.concat %[[A]], %[[A]] : vector<8xi32>, vector<16xi32>
    // CHECK: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK: %[[MAC:.*]] = aievec_aie1.mac %[[AA]], %[[B]], %[[ACC]] : vector<16xi32>, vector<8xi32>, vector<8xi80>
    // CHECK: %[[RES:.*]] = aievec.srs %[[MAC]], %[[C0]] : vector<8xi80>, i32, vector<8xi32>
    %0 = arith.muli %a, %b : vector<8xi32>
    %1 = arith.addi %c, %0 : vector<8xi32>
    // CHECK: return %[[RES]] : vector<8xi32>
    return %1 : vector<8xi32>
}

// CHECK-LABEL: func.func @splatAndMac2SplatMac(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @splatAndMac2SplatMac(%a : vector<8xi32>,
                                %b : vector<8xi32>,
                                %c : vector<8xi32>) -> vector<8xi80> {
    %0 = vector.extract %a[2] : i32 from vector<8xi32>
    %1 = vector.broadcast %0 : i32 to vector<8xi32>
    %2 = aievec.concat %1, %1 : vector<8xi32>, vector<16xi32>
    %3 = aievec.ups %c {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    %4 = aievec_aie1.mac %2, %b, %3 : vector<16xi32>, vector<8xi32>, vector<8xi80>
    return %4 : vector<8xi80>
    // CHECK-DAG: %[[CVZ:.*]] = arith.constant dense<0> : vector<8xi32>
    // CHECK-DAG: %[[BB:.*]] = aievec.concat %[[B]], %[[CVZ]] : vector<8xi32>, vector<16xi32>
    // CHECK-DAG: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK-DAG: %[[MAC:.*]] = aievec_aie1.mac %[[BB]], %[[A]], %[[ACC]]
    // CHECK-SAME:              {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000",
    // CHECK-SAME:               zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
    // CHECK: return %[[MAC]] : vector<8xi80>
}

// CHECK-LABEL: func.func @splatAndMac2SplatMac_inv(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @splatAndMac2SplatMac_inv(%a : vector<8xi32>,
                                    %b : vector<8xi32>,
                                    %c : vector<8xi32>) -> vector<8xi80> {
    %0 = vector.extract %b[4] : i32 from vector<8xi32>
    %1 = vector.broadcast %0 : i32 to vector<8xi32>
    %2 = aievec.concat %a, %a : vector<8xi32>, vector<16xi32>
    %3 = aievec.ups %c {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    %4 = aievec_aie1.mac %2, %1, %3 : vector<16xi32>, vector<8xi32>, vector<8xi80>
    return %4 : vector<8xi80>
    // CHECK-DAG: %[[CVZ:.*]] = arith.constant dense<0> : vector<8xi32>
    // CHECK-DAG: %[[AA:.*]] = aievec.concat %[[A]], %[[CVZ]] : vector<8xi32>, vector<16xi32>
    // CHECK-DAG: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK-DAG: %[[MAC:.*]] = aievec_aie1.mac %[[AA]], %[[B]], %[[ACC]]
    // CHECK-SAME:              {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000",
    // CHECK-SAME:               zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
    // CHECK: return %[[MAC]] : vector<8xi80>
}

// CHECK-LABEL: func.func @splatAndMac2SplatMacI16(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<16xi16>
func.func @splatAndMac2SplatMacI16(%a : vector<16xi16>,
                                   %b : vector<16xi16>,
                                   %c : vector<16xi16>) -> vector<16xi48> {
    %0 = vector.extract %a[3] : i16 from vector<16xi16>
    %1 = vector.broadcast %0 : i16 to vector<16xi16>
    %2 = aievec.concat %1, %1 : vector<16xi16>, vector<32xi16>
    %3 = aievec.ups %c {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
    %4 = aievec_aie1.mac %2, %b, %3 : vector<32xi16>, vector<16xi16>, vector<16xi48>
    return %4 : vector<16xi48>
    // CHECK-DAG: %[[CVZ:.*]] = arith.constant dense<0> : vector<16xi16>
    // CHECK-DAG: %[[BB:.*]] = aievec.concat %[[B]], %[[CVZ]] : vector<16xi16>, vector<32xi16>
    // CHECK-DAG: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
    // CHECK-DAG: %[[MAC:.*]] = aievec_aie1.mac %[[BB]], %[[A]], %[[ACC]]
    // CHECK-SAME:              {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
    // CHECK-SAME:               xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "3", zstep = "1"}
    // CHECK-SAME:              : vector<32xi16>, vector<16xi16>, vector<16xi48>
    // CHECK: return %[[MAC]] : vector<16xi48>
}
