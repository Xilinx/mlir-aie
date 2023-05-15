// RUN: aie-translate %s -aievec-to-cpp | FileCheck %s


// CHECK: int32_t external_function(v16int32);
func.func private @external_function(%v : vector<16xi32>) -> i32

// CHECK: void external_function_with_memref(int16_t * restrict);
func.func private @external_function_with_memref(%m : memref<64xi16>)

// CHECK: void external_function_with_dynamic_memref(int8_t * restrict, size_t);
func.func private @external_function_with_dynamic_memref(%m : memref<?xi8>)
