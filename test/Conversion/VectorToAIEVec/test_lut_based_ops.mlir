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

// CHECK-LABEL: func @test_exp_lut_v32bf16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xbf16>
func.func @test_exp_lut_v32bf16(%a: vector<32xbf16>) -> vector<32xbf16> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // Extract low and high halves
    // CHECK: %[[LOW:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
    // CHECK: %[[HIGH:.*]] = aievec.ext %[[A]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
    // Process low half: call -> bitcast -> srs
    // CHECK: %[[CALL_LOW:.*]] = call @getExpBf16(%[[LOW]]) : (vector<16xbf16>) -> vector<8xi64>
    // CHECK: %[[CAST_LOW:.*]] = vector.bitcast %[[CALL_LOW]] : vector<8xi64> to vector<16xf32>
    // CHECK: %[[SRS_LOW:.*]] = aievec.srs %[[CAST_LOW]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
    // Process high half: call -> bitcast -> srs
    // CHECK: %[[CALL_HIGH:.*]] = call @getExpBf16(%[[HIGH]]) : (vector<16xbf16>) -> vector<8xi64>
    // CHECK: %[[CAST_HIGH:.*]] = vector.bitcast %[[CALL_HIGH]] : vector<8xi64> to vector<16xf32>
    // CHECK: %[[SRS_HIGH:.*]] = aievec.srs %[[CAST_HIGH]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
    // Concat back together
    // CHECK: %[[RESULT:.*]] = aievec.concat %[[SRS_LOW]], %[[SRS_HIGH]] : vector<16xbf16>, vector<32xbf16>
    %0 = math.exp %a : vector<32xbf16>
    // CHECK: return %[[RESULT]] : vector<32xbf16>
    return %0 : vector<32xbf16>
}
