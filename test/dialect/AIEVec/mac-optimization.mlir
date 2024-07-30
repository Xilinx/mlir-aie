// RUN: aie-opt %s -optimize-aievec -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @merge_single_column_mac(
// CHECK-SAME: %[[VA:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[VB:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[VC:[A-Za-z0-9]+]]: vector<16xi16>) -> vector<16xi48> {
func.func @merge_single_column_mac(%A : vector<16xi16>,
                                   %B : vector<16xi16>,
                                   %C : vector<16xi16>) -> vector<16xi48> {
    // CHECK: %[[ACC:.*]] = arith.constant dense<0> : vector<16xi48>
    // CHECK-NEXT: %[[VAB:.*]] = aievec.concat %[[VA]], %[[VB]] : vector<16xi16>, vector<32xi16>
    // CHECK-NEXT: %[[MAC:.*]] = aievec_aie1.mac %[[VAB]], %[[VC]], %[[ACC]] {
    // CHECK-SAME: xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120", xstart = "0",
    // CHECK-SAME: zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"}
    // CHECK-SAME: : vector<32xi16>, vector<16xi16>, vector<16xi48>
    // CHECK-NEXT: return %[[MAC]] : vector<16xi48>
    %acc = arith.constant dense<0> : vector<16xi48>
    %zvec = arith.constant dense<0> : vector<16xi16>
    %la = aievec.concat %A, %zvec : vector<16xi16>, vector<32xi16>
    %mac0 = aievec_aie1.mac %la, %C, %acc {xoffsets = "0x73727170",
                                      xoffsets_hi = "0x77767574",
                                      xsquare = "0x3120", xstart = "0",
                                      zoffsets = "0", zoffsets_hi = "0",
                                      zstart = "0", zstep = "1"}
                                    : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %lb = aievec.concat %B, %zvec : vector<16xi16>, vector<32xi16>
    %mac1 = aievec_aie1.mac %lb, %C, %mac0 {xoffsets = "0x73727170",
                                       xoffsets_hi = "0x77767574",
                                       xsquare = "0x3120", xstart = "0",
                                       zoffsets = "0", zoffsets_hi = "0",
                                       zstart = "1", zstep = "1"}
                                    : vector<32xi16>, vector<16xi16>, vector<16xi48>
    return %mac1 : vector<16xi48>
}

// -----

// CHECK-LABEL: func.func @merge_single_column_mac(
// CHECK-SAME: %[[VA:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[VB:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[VC:[A-Za-z0-9]+]]: vector<16xi16>) -> vector<16xi48> {
func.func @merge_single_column_mac(%A : vector<16xi16>,
                                   %B : vector<16xi16>,
                                   %C : vector<16xi16>) -> vector<16xi48> {
    // CHECK: %[[ACC:.*]] = arith.constant dense<0> : vector<16xi48>
    // CHECK-NEXT: %[[VAB:.*]] = aievec.concat %[[VA]], %[[VB]] : vector<16xi16>, vector<32xi16>
    // CHECK-NEXT: %[[MAC:.*]] = aievec_aie1.mac %[[VAB]], %[[VC]], %[[ACC]] {
    // CHECK-SAME: xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120", xstart = "0",
    // CHECK-SAME: zoffsets = "0", zoffsets_hi = "0", zstart = "3", zstep = "4"}
    // CHECK-SAME: : vector<32xi16>, vector<16xi16>, vector<16xi48>
    // CHECK-NEXT: return %[[MAC]] : vector<16xi48>
    %acc = arith.constant dense<0> : vector<16xi48>
    %zvec = arith.constant dense<0> : vector<16xi16>
    %la = aievec.concat %A, %zvec : vector<16xi16>, vector<32xi16>
    %mac0 = aievec_aie1.mac %la, %C, %acc {xoffsets = "0x73727170",
                                      xoffsets_hi = "0x77767574",
                                      xsquare = "0x3120", xstart = "0",
                                      zoffsets = "0", zoffsets_hi = "0",
                                      zstart = "3", zstep = "1"}
                                    : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %lb = aievec.concat %B, %zvec : vector<16xi16>, vector<32xi16>
    %mac1 = aievec_aie1.mac %lb, %C, %mac0 {xoffsets = "0x73727170",
                                       xoffsets_hi = "0x77767574",
                                       xsquare = "0x3120", xstart = "0",
                                       zoffsets = "0", zoffsets_hi = "0",
                                       zstart = "7", zstep = "1"}
                                    : vector<32xi16>, vector<16xi16>, vector<16xi48>
    return %mac1 : vector<16xi48>
}

// -----

// CHECK-LABEL: func.func @merge_single_column_mac(
// CHECK-SAME: %[[VA:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[VB:[A-Za-z0-9]+]]: vector<16xi16>,
// CHECK-SAME: %[[VC:[A-Za-z0-9]+]]: vector<16xi16>) -> vector<16xi48> {
func.func @merge_single_column_mac(%A : vector<16xi16>,
                                   %B : vector<16xi16>,
                                   %C : vector<16xi16>) -> vector<16xi48> {
    // CHECK: %[[ACC:.*]] = arith.constant dense<0> : vector<16xi48>
    // CHECK-NEXT: %[[VBA:.*]] = aievec.concat %[[VB]], %[[VA]] : vector<16xi16>, vector<32xi16>
    // CHECK-NEXT: %[[MAC:.*]] = aievec_aie1.mac %[[VBA]], %[[VC]], %[[ACC]] {
    // CHECK-SAME: xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120", xstart = "0",
    // CHECK-SAME: zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "5"}
    // CHECK-SAME: : vector<32xi16>, vector<16xi16>, vector<16xi48>
    // CHECK-NEXT: return %[[MAC]] : vector<16xi48>
    %acc = arith.constant dense<0> : vector<16xi48>
    %zvec = arith.constant dense<0> : vector<16xi16>
    %la = aievec.concat %A, %zvec : vector<16xi16>, vector<32xi16>
    %mac0 = aievec_aie1.mac %la, %C, %acc {xoffsets = "0x73727170",
                                      xoffsets_hi = "0x77767574",
                                      xsquare = "0x3120", xstart = "0",
                                      zoffsets = "0", zoffsets_hi = "0",
                                      zstart = "9", zstep = "1"}
                                    : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %lb = aievec.concat %B, %zvec : vector<16xi16>, vector<32xi16>
    %mac1 = aievec_aie1.mac %lb, %C, %mac0 {xoffsets = "0x73727170",
                                       xoffsets_hi = "0x77767574",
                                       xsquare = "0x3120", xstart = "0",
                                       zoffsets = "0", zoffsets_hi = "0",
                                       zstart = "4", zstep = "1"}
                                    : vector<32xi16>, vector<16xi16>, vector<16xi48>
    return %mac1 : vector<16xi48>
}
