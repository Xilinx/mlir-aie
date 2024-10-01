// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" | FileCheck %s

// CHECK-LABEL: func private @getTanhBf16(vector<16xbf16>) -> vector<16xbf16>
// CHECK-LABEL: func @test_tanh_lut
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
module{
func.func @test_tanh_lut(%a: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[CALL:.*]] = call @getTanhBf16(%[[A]]) : (vector<16xbf16>) -> vector<16xbf16>
    %0 = math.tanh %a : vector<16xbf16>
    // CHECK: return %[[CALL]] : vector<16xbf16>
    return %0 : vector<16xbf16>
}
}
