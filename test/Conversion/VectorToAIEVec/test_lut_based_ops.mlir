// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" | FileCheck %s


// CHECK-LABEL: func private @getExpBf16(vector<16xbf16>) -> vector<8xi64>
// CHECK-LABEL: func @test_exp_lut
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_exp_lut(%a: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[CALL:.*]] = call @getExpBf16(%[[A]]) : (vector<16xbf16>) -> vector<8xi64>
    // CHECK: %[[CAST:.*]] = vector.bitcast %[[CALL]] : vector<8xi64> to vector<16xf32>
    // CHECK: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
    %0 = math.exp %a : vector<16xbf16>
    // CHECK: return %[[SRS]] : vector<16xbf16>
    return %0 : vector<16xbf16>
}
