# AIE Automatic Vectorization

Achieving high performance with the AIEngine architecture typically requires executing vector multiply-accumulate instructions.  Ideally, an algorithm would start a new pipelined vector multiple-accumulate instruction on every clock cycle, although in practice achieving this outside of high-performance inner loops is very difficult.  Explicitly coding vector operations in C++ can take significant expertise, taking into account memory layout, register pressure, and different data types. [Ref:Vyasa](https://github.com/stephenneuendorffer/vyasa)  Fortunately, by leveraging the MLIR infrastructure, it is possible to generate reasonably high-performance code in many circumstances.

## Affine Loops

A typical starting point for automatic vectorization is to consider Affine looped programs. [Example](https://github.com/Xilinx/mlir-aie/tree/main/test/aievec/pointwise_mult_f32.mlir)
```
func.func @pointwise_mult (%A: memref<2048xf32>, %B: memref<2048xf32>, %C: memref<2048xf32>) {
    affine.for %arg0 = 0 to 2048 {
       %a = affine.load %A[%arg0] : memref<2048xf32>
       %b = affine.load %B[%arg0] : memref<2048xf32>
       %c = arith.mulf %a, %b : f32
       affine.store %c, %C[%arg0] : memref<2048xf32>
    }
    return
}
```

This code is typical of looped code that vectorizes easily, since the loop accesses data in order and every block of outputs corresponds to a separate vector of inputs.  In addition, the loop is an exact multiple of the vector size, so each block is complete.  This makes it a good introductory example, but the steps below can also be effectively applied to more complicated examples.

## Generic Vectorization

The ["Affine Super-Vectorize" pass](https://mlir.llvm.org/docs/Passes/#-affine-super-vectorize-vectorize-to-a-target-independent-n-d-vector-abstraction) in MLIR Upstream is capable of extracting generic vector operations from the above code.  In order to target the AIEngine 8-wide vector floating point unit, we select a vector size of 8:
```
aie-opt -affine-super-vectorize="virtual-vector-size=8 vectorize-reductions" < pointwise_mult_f32.mlir
```
```
  func.func @pointwise_mult(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>, %arg2: memref<2048xf32>) {
    affine.for %arg3 = 0 to 2048 step 8 {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = vector.transfer_read %arg0[%arg3], %cst : memref<2048xf32>, vector<8xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %1 = vector.transfer_read %arg1[%arg3], %cst_0 : memref<2048xf32>, vector<8xf32>
      %2 = arith.mulf %0, %1 : vector<8xf32>
      vector.transfer_write %2, %arg2[%arg3] : vector<8xf32>, memref<2048xf32>
    }
    return
  }
```

The super-vectorizer blocks the loop by a factor of 8, replacing scalar operations with corresponding vector operations.

## AIEngine-specific vector operations

The ['AIE Vectorize' pass](https://xilinx.github.io/mlir-aie/AIEVecPasses.html) in this repository transforms the above vector code into AIEngine-specific vector operations, represented in the [AIEVec Dialect](https://xilinx.github.io/mlir-aie/AIEVecOps.html).  These operations use types that are directly implementable in the AIEngine architecture and represent device specific features, such as the vector permute network.

```
aie-opt -affine-super-vectorize="virtual-vector-size=8 vectorize-reductions" --aie-vectorize < pointwise_mult_f32.mlir
```
```
  func.func @pointwise_mult(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>, %arg2: memref<2048xf32>) {
    %c8 = arith.constant 8 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c2048 step %c8 {
      %0 = aievec.upd %arg0[%arg3] {index = 0 : i8, offset = 0 : si32} : memref<2048xf32>, vector<8xf32>
      %1 = aievec.upd %arg1[%arg3] {index = 0 : i8, offset = 0 : si32} : memref<2048xf32>, vector<8xf32>
      %2 = aievec.concat %0, %0 : vector<8xf32>, vector<16xf32>
      %3 = aievec.mul %2, %1 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x76543210", zstart = "0"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
      %4 = aievec.srs %3 {shift = 0 : i8} : !aievec.acc<8xf32>, vector<8xf32>
      vector.transfer_write %4, %arg2[%arg3] {in_bounds = [true]} : vector<8xf32>, memref<2048xf32>
    }
    return
  }
```

This code can be translated to C++ code that can be included in a Vitis design:
```
aie-opt -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize < ../../aie/test/aievec/pointwise_mult_f32.mlir | aie-translate --aievec-to-cpp
```
```
void pointwise_mult(float * restrict v1, float * restrict v2, float * restrict v3) {
  size_t v4 = 0;
  size_t v5 = 2048;
  size_t v6 = 8;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
  chess_prepare_for_pipelining
  chess_loop_range(256, 256)
  {
    v8float v8 = *(v8float *)(v1 + v7);
    v8float v9 = *(v8float *)(v2 + v7);
    v16float v10 = concat(v8, v8);
    v8float v11 = fpmul(v10, 0, 0x76543210, v9, 0, 0x76543210);
    *(v8float *)(v3 + v7) = v11;
  }
  return;
}
```

## Vectorizing integer types

The AIEngine architecture supports a number of different datatypes, typically supporting different vector sizes.  For 16-bit values we can vectorize with 
```
aie-opt -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize < pointwise_mult_i16.mlir
```
```
func.func @pointwise_mult (%A: memref<2048xi16>, %B: memref<2048xi16>, %C: memref<2048xi16>) {
    affine.for %arg0 = 0 to 2048 {
       %a = affine.load %A[%arg0] : memref<2048xi16>
       %b = affine.load %B[%arg0] : memref<2048xi16>
       %c = arith.muli %a, %b : i16
       affine.store %c, %C[%arg0] : memref<2048xi16>
    }
    return
}
```
Results in:
```
  func.func @pointwise_mult(%arg0: memref<2048xi16>, %arg1: memref<2048xi16>, %arg2: memref<2048xi16>) {
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c16 = arith.constant 16 : index
    scf.for %arg3 = %c0 to %c2048 step %c16 {
      %0 = aievec.upd %arg0[%arg3] {index = 0 : i8, offset = 0 : si32} : memref<2048xi16>, vector<16xi16>
      %1 = aievec.upd %arg1[%arg3] {index = 0 : i8, offset = 0 : si32} : memref<2048xi16>, vector<16xi16>
      %2 = aievec.mul %0, %1 : vector<16xi16>, vector<16xi16>, !aievec.acc<16xi48>
      %3 = aievec.srs %2 {shift = 0 : i8} : !aievec.acc<16xi48>, vector<16xi16>
      vector.transfer_write %3, %arg2[%arg3] {in_bounds = [true]} : vector<16xi16>, memref<2048xi16>
    }
    return
  }
```

## Vectorizing Convolutions

More complex algorithms with multiple loops can be more challenging to vectorize.  Finding a good vectorization scheme may require exploring a number of different vectorization possibilities.  Often it is beneficial to unroll inner loops whose bounds are too small to vectorize.  For instance, in a 2-D convolution, typical in machine learning, unrolling small loops results in a good vectorization strategy:

```
aie-opt --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize < conv2d_i32.mlir
```
```
  func.func @conv2d(%arg0: memref<2048x2048xi32>, %arg1: memref<9xi32>, %arg2: memref<2046x2046xi32>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<9xi32>, vector<8xi32>
    %1 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : si32} : memref<9xi32>, vector<8xi32>
    %c0_0 = arith.constant 0 : index
    %c2046 = arith.constant 2046 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0_0 to %c2046 step %c1 {
      %c1_1 = arith.constant 1 : index
      %2 = arith.addi %arg3, %c1_1 : index
      %c2 = arith.constant 2 : index
      %3 = arith.addi %arg3, %c2 : index
      %c0_2 = arith.constant 0 : index
      %c2046_3 = arith.constant 2046 : index
      %c8_4 = arith.constant 8 : index
      scf.for %arg4 = %c0_2 to %c2046_3 step %c8_4 {
        %4 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<2046x2046xi32>, vector<8xi32>
        %5 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<2048x2048xi32>, vector<16xi32>
        %6 = aievec.ups %4 {shift = 0 : i8} : vector<8xi32>, !aievec.acc<8xi80>
        %7 = aievec.mac %5, %0, %6 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %c1_5 = arith.constant 1 : index
        %8 = arith.addi %arg4, %c1_5 : index
        %9 = aievec.upd %arg0[%arg3, %8], %5 {index = 1 : i8, offset = 224 : si32} : memref<2048x2048xi32>, vector<16xi32>
        %10 = aievec.mac %9, %0, %7 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %11 = aievec.mac %9, %0, %10 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %12 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<2048x2048xi32>, vector<16xi32>
        %13 = aievec.mac %12, %0, %11 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %14 = aievec.upd %arg0[%2, %8], %12 {index = 1 : i8, offset = 224 : si32} : memref<2048x2048xi32>, vector<16xi32>
        %15 = aievec.mac %14, %0, %13 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %16 = aievec.mac %14, %0, %15 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %17 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<2048x2048xi32>, vector<16xi32>
        %18 = aievec.mac %17, %0, %16 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %19 = aievec.upd %arg0[%3, %8], %17 {index = 1 : i8, offset = 224 : si32} : memref<2048x2048xi32>, vector<16xi32>
        %20 = aievec.mac %19, %0, %18 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %21 = aievec.mac %19, %1, %20 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
        %22 = aievec.srs %21 {shift = 0 : i8} : !aievec.acc<8xi80>, vector<8xi32>
        vector.transfer_write %22, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xi32>, memref<2046x2046xi32>
      }
    }
    return
  }
  ```

## Starting with C code using Polygeist

One way to generate the code for Affine loops is using [Polygeist](https://github.com/wsmoses/Polygeist), a clang-based frontend to MLIR.

```
mlir-clang conv2d.c --function=conv2d -S --raise-scf-to-affine
```
```
void conv2d(int img_in[17][272], int kernel_coeff[3][3], int img_out[16][256]) {
    for(int r = 0; r < 16; r++)
        for(int c = 0; c < 256; c++) {
            int acc = 0;
            for(int i = 0; i < 3; i++)
                for(int j = 0; j < 3; j++) {
                    acc += img_in[r+i][c+j] * kernel_coeff[i][j];
                }
            img_out[r][c] = acc;
        }
}
```

Resulting in
```
  func.func @conv2d(%arg0: memref<?x272xi32>, %arg1: memref<?x3xi32>, %arg2: memref<?x256xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 256 {
        %0 = affine.for %arg5 = 0 to 3 iter_args(%arg6 = %c0_i32) -> (i32) {
          %1 = affine.for %arg7 = 0 to 3 iter_args(%arg8 = %arg6) -> (i32) {
            %2 = affine.load %arg0[%arg3 + %arg5, %arg4 + %arg7] : memref<?x272xi32>
            %3 = affine.load %arg1[%arg5, %arg7] : memref<?x3xi32>
            %4 = arith.muli %2, %3 : i32
            %5 = arith.addi %arg8, %4 : i32
            affine.yield %5 : i32
          }
          affine.yield %1 : i32
        }
        affine.store %0, %arg2[%arg3, %arg4] : memref<?x256xi32>
      }
    }
    return
  }
```

Running the whole pipeline, we get:
```
mlir-clang --function=conv2d conv2d_i32.c -S --raise-scf-to-affine | aie-opt --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=8 vectorize-reductions" --aie-vectorize | aie-translate --aievec-to-cpp
```
```
mlir-clang --function=conv2d conv2d_i32.c -S --raise-scf-to-affine | aie-opt --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=8 vectorize-reductions" --aie-vectorize | aie-translate --aievec-to-cpp
void conv2d(int32_t * restrict v4, size_t m1, int32_t * restrict v5, size_t m2, int32_t * restrict v6, size_t m3) {
  size_t v7 = 0;
  size_t v8 = 2;
  v8int32 v9 = *(v8int32 *)(v5 + 3*v7+v7);
  v8int32 v10 = *(v8int32 *)(v5 + 3*v8+v8);
  size_t v11 = 0;
  size_t v12 = 16;
  size_t v13 = 1;
  for (size_t v14 = v11; v14 < v12; v14 += v13)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v15 = 1;
    size_t v16 = v14 + v15;
    size_t v17 = 2;
    size_t v18 = v14 + v17;
    size_t v19 = 0;
    size_t v20 = 256;
    size_t v21 = 8;
    for (size_t v22 = v19; v22 < v20; v22 += v21)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v16int32 v23;
      int32_t * restrict r_v23_v4 = v4;
      v23 = upd_w(v23, 0, *(v8int32 *)(r_v23_v4 + 272*v14+v22));
      v8acc80 v24 = lmul8(v23, 0, 0x76543210, v9, 0, 0x00000000);
      size_t v25 = 1;
      size_t v26 = v22 + v25;
      v23 = upd_w(v23, 1, *(v8int32 *)(r_v23_v4 + 272*v14+v26 + 7));
      v24 = lmac8(v24, v23, 1, 0x76543210, v9, 1, 0x00000000);
      v24 = lmac8(v24, v23, 2, 0x76543210, v9, 2, 0x00000000);
      v16int32 v27;
      int32_t * restrict r_v27_v4 = v4;
      v27 = upd_w(v27, 0, *(v8int32 *)(r_v27_v4 + 272*v16+v22));
      v24 = lmac8(v24, v27, 0, 0x76543210, v9, 3, 0x00000000);
      v27 = upd_w(v27, 1, *(v8int32 *)(r_v27_v4 + 272*v16+v26 + 7));
      v24 = lmac8(v24, v27, 1, 0x76543210, v9, 4, 0x00000000);
      v24 = lmac8(v24, v27, 2, 0x76543210, v9, 5, 0x00000000);
      v16int32 v28;
      int32_t * restrict r_v28_v4 = v4;
      v28 = upd_w(v28, 0, *(v8int32 *)(r_v28_v4 + 272*v18+v22));
      v24 = lmac8(v24, v28, 0, 0x76543210, v9, 6, 0x00000000);
      v28 = upd_w(v28, 1, *(v8int32 *)(r_v28_v4 + 272*v18+v26 + 7));
      v24 = lmac8(v24, v28, 1, 0x76543210, v9, 7, 0x00000000);
      v24 = lmac8(v24, v28, 2, 0x76543210, v10, 0, 0x00000000);
      v8int32 v29 = srs(v24, 0);
      *(v8int32 *)(v6 + 256*v14+v22) = v29;
    }
  }
  return;
}
```