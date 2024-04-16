module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %cst_2 = arith.constant dense<0xFF80> : vector<32xbf16>
    %0 = affine.for %arg2 = 0 to 1024 step 32 iter_args(%arg3 = %cst_2) -> (vector<32xbf16>) {
      %5 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>, vector<32xbf16>
      %6 = arith.maximumf %arg3, %5 : vector<32xbf16>
      affine.yield %6 : vector<32xbf16>
    }
    %1 = vector.reduction <maximumf>, %0 : vector<32xbf16> into bf16
    affine.for %arg2 = 0 to 1024 {
      %5 = affine.load %arg0[%arg2] : memref<1024xbf16>
      %6 = arith.subf %5, %1 : bf16
      %7 = math.exp %6 : bf16
      affine.store %7, %arg0[%arg2] : memref<1024xbf16>
    }
    %2 = affine.for %arg2 = 0 to 1024 iter_args(%arg3 = %cst) -> (f32) {
      %5 = affine.load %arg0[%arg2] : memref<1024xbf16>
      %6 = arith.extf %5 : bf16 to f32
      %7 = arith.addf %arg3, %6 : f32
      affine.yield %7 : f32
    }
    %3 = arith.divf %cst_0, %2 : f32
    %4 = arith.truncf %3 : f32 to bf16
    affine.for %arg2 = 0 to 1024 {
      %5 = affine.load %arg0[%arg2] : memref<1024xbf16>
      %6 = arith.mulf %5, %4 : bf16
      affine.store %6, %arg1[%arg2] : memref<1024xbf16>
    }
    return
  }
}