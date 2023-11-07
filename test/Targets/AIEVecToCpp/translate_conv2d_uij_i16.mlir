// RUN: aie-translate --aievec-to-cpp %s -split-input-file | FileCheck %s

// CHECK-LABEL: void conv2d(int16_t * restrict v1, int16_t * restrict v2, int16_t * restrict v3) {
module  {
  func.func @conv2d(%arg0: memref<2048x2048xi16>, %arg1: memref<12xi16>, %arg2: memref<2046x2046xi16>) {
    %c0 = arith.constant 0 : index
    %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<12xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c2046 = arith.constant 2046 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0_0 to %c2046 step %c1 {
      %c1_1 = arith.constant 1 : index
      %1 = arith.addi %arg3, %c1_1 : index
      %c2 = arith.constant 2 : index
      %2 = arith.addi %arg3, %c2 : index
      %c0_2 = arith.constant 0 : index
      %c2046_3 = arith.constant 2046 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0_2 to %c2046_3 step %c16 {
        %3 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2046x2046xi16>, vector<16xi16>
        %4 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %5 = aievec.ups %3 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %6 = aievec.mac %4, %0, %5 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %c1_4 = arith.constant 1 : index
        %7 = arith.addi %arg4, %c1_4 : index
        %8 = aievec.upd %arg0[%arg3, %7], %4 {index = 1 : i8, offset = 240 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %9 = aievec.mac %8, %0, %6 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec.upd %arg0[%1, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %11 = aievec.mac %10, %0, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %12 = aievec.upd %arg0[%1, %7], %10 {index = 1 : i8, offset = 240 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %13 = aievec.mac %12, %0, %11 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %15 = aievec.mac %14, %0, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %16 = aievec.upd %arg0[%2, %7], %14 {index = 1 : i8, offset = 240 : i32} : memref<2048x2048xi16>, vector<32xi16>
        %17 = aievec.mac %16, %0, %15 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec.srs %17 {shift = 0 : i8} : vector<16xi48>, vector<16xi16>
        vector.transfer_write %18, %arg2[%arg3, %arg4] : vector<16xi16>, memref<2046x2046xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v4 = 0;
//CHECK-NEXT:  v16int16 v5 = *(v16int16 *)(v2 + v4);
//CHECK-NEXT:  size_t v6 = 0;
//CHECK-NEXT:  size_t v7 = 2046;
//CHECK-NEXT:  size_t v8 = 1;
//CHECK-NEXT:  for (size_t v9 = v6; v9 < v7; v9 += v8)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  chess_loop_range(2046, 2046)
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v10 = 1;
//CHECK-NEXT:    size_t v11 = v9 + v10;
//CHECK-NEXT:    size_t v12 = 2;
//CHECK-NEXT:    size_t v13 = v9 + v12;
//CHECK-NEXT:    size_t v14 = 0;
//CHECK-NEXT:    size_t v15 = 2046;
//CHECK-NEXT:    size_t v16 = 16;
//CHECK-NEXT:    for (size_t v17 = v14; v17 < v15; v17 += v16)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(127, 128)
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v18 = *(v16int16 *)(v3 + 2046*v9+v17);
//CHECK-NEXT:      v32int16 v19;
//CHECK-NEXT:      int16_t * restrict r_v19_v1 = v1;
//CHECK-NEXT:      v19 = upd_w(v19, 0, *(v16int16 *)(r_v19_v1 + 2048*v9+v17));
//CHECK-NEXT:      v16acc48 v20 = ups(v18, 0);
//CHECK-NEXT:      v20 = mac16(v20, v19, 0, 0x03020100, 0x07060504, 0x2110, v5, 0, 0, 0, 1);
//CHECK-NEXT:      size_t v21 = 1;
//CHECK-NEXT:      size_t v22 = v17 + v21;
//CHECK-NEXT:      v19 = upd_w(v19, 1, *(v16int16 *)(r_v19_v1 + 2048*v9+v22 + 15));
//CHECK-NEXT:      v20 = mac16(v20, v19, 2, 0x03020100, 0x07060504, 0x2110, v5, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v23;
//CHECK-NEXT:      int16_t * restrict r_v23_v1 = v1;
//CHECK-NEXT:      v23 = upd_w(v23, 0, *(v16int16 *)(r_v23_v1 + 2048*v11+v17));
//CHECK-NEXT:      v20 = mac16(v20, v23, 0, 0x03020100, 0x07060504, 0x2110, v5, 4, 0, 0, 1);
//CHECK-NEXT:      v23 = upd_w(v23, 1, *(v16int16 *)(r_v23_v1 + 2048*v11+v22 + 15));
//CHECK-NEXT:      v20 = mac16(v20, v23, 2, 0x03020100, 0x07060504, 0x2110, v5, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v24;
//CHECK-NEXT:      int16_t * restrict r_v24_v1 = v1;
//CHECK-NEXT:      v24 = upd_w(v24, 0, *(v16int16 *)(r_v24_v1 + 2048*v13+v17));
//CHECK-NEXT:      v20 = mac16(v20, v24, 0, 0x03020100, 0x07060504, 0x2110, v5, 8, 0, 0, 1);
//CHECK-NEXT:      v24 = upd_w(v24, 1, *(v16int16 *)(r_v24_v1 + 2048*v13+v22 + 15));
//CHECK-NEXT:      v20 = mac16(v20, v24, 2, 0x03020100, 0x07060504, 0x2110, v5, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v25 = srs(v20, 0);
//CHECK-NEXT:      *(v16int16 *)(v3 + 2046*v9+v17) = v25;
//CHECK-NEXT:    }
//CHECK-NEXT:  }


// CHECK-LABEL: void conv2d(int16_t * restrict v6, size_t m1, size_t m2, int16_t * restrict v7, size_t m3, int16_t * restrict v8, size_t m4, size_t m5, size_t v9, size_t v10) {
module  {
  func.func @conv2d(%arg0: memref<?x?xi16>, %arg1: memref<?xi16>, %arg2: memref<?x?xi16>, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0_0 to %arg3 step %c1 {
      %c1_1 = arith.constant 1 : index
      %1 = arith.addi %arg5, %c1_1 : index
      %c2 = arith.constant 2 : index
      %2 = arith.addi %arg5, %c2 : index
      %c0_2 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      scf.for %arg6 = %c0_2 to %arg4 step %c16 {
        %3 = aievec.upd %arg2[%arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<16xi16>
        %4 = aievec.upd %arg0[%arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %5 = aievec.ups %3 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %6 = aievec.mac %4, %0, %5 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %c1_3 = arith.constant 1 : index
        %7 = arith.addi %arg6, %c1_3 : index
        %8 = aievec.upd %arg0[%arg5, %7], %4 {index = 1 : i8, offset = 240 : i32} : memref<?x?xi16>, vector<32xi16>
        %9 = aievec.mac %8, %0, %6 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec.upd %arg0[%1, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %11 = aievec.mac %10, %0, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %12 = aievec.upd %arg0[%1, %7], %10 {index = 1 : i8, offset = 240 : i32} : memref<?x?xi16>, vector<32xi16>
        %13 = aievec.mac %12, %0, %11 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec.upd %arg0[%2, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %15 = aievec.mac %14, %0, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %16 = aievec.upd %arg0[%2, %7], %14 {index = 1 : i8, offset = 240 : i32} : memref<?x?xi16>, vector<32xi16>
        %17 = aievec.mac %16, %0, %15 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec.srs %17 {shift = 0 : i8} : vector<16xi48>, vector<16xi16>
        vector.transfer_write %18, %arg2[%arg5, %arg6] : vector<16xi16>, memref<?x?xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v11 = 0;
//CHECK-NEXT:  v16int16 v12 = *(v16int16 *)(v7 + v11);
//CHECK-NEXT:  size_t v13 = 0;
//CHECK-NEXT:  size_t v14 = 1;
//CHECK-NEXT:  for (size_t v15 = v13; v15 < v9; v15 += v14)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v16 = 1;
//CHECK-NEXT:    size_t v17 = v15 + v16;
//CHECK-NEXT:    size_t v18 = 2;
//CHECK-NEXT:    size_t v19 = v15 + v18;
//CHECK-NEXT:    size_t v20 = 0;
//CHECK-NEXT:    size_t v21 = 16;
//CHECK-NEXT:    for (size_t v22 = v20; v22 < v10; v22 += v21)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v23 = *(v16int16 *)(v8 + m5*v15+v22);
//CHECK-NEXT:      v32int16 v24;
//CHECK-NEXT:      int16_t * restrict r_v24_v6 = v6;
//CHECK-NEXT:      v24 = upd_w(v24, 0, *(v16int16 *)(r_v24_v6 + m2*v15+v22));
//CHECK-NEXT:      v16acc48 v25 = ups(v23, 0);
//CHECK-NEXT:      v25 = mac16(v25, v24, 0, 0x03020100, 0x07060504, 0x2110, v12, 0, 0, 0, 1);
//CHECK-NEXT:      size_t v26 = 1;
//CHECK-NEXT:      size_t v27 = v22 + v26;
//CHECK-NEXT:      v24 = upd_w(v24, 1, *(v16int16 *)(r_v24_v6 + m2*v15+v27 + 15));
//CHECK-NEXT:      v25 = mac16(v25, v24, 2, 0x03020100, 0x07060504, 0x2110, v12, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v28;
//CHECK-NEXT:      int16_t * restrict r_v28_v6 = v6;
//CHECK-NEXT:      v28 = upd_w(v28, 0, *(v16int16 *)(r_v28_v6 + m2*v17+v22));
//CHECK-NEXT:      v25 = mac16(v25, v28, 0, 0x03020100, 0x07060504, 0x2110, v12, 4, 0, 0, 1);
//CHECK-NEXT:      v28 = upd_w(v28, 1, *(v16int16 *)(r_v28_v6 + m2*v17+v27 + 15));
//CHECK-NEXT:      v25 = mac16(v25, v28, 2, 0x03020100, 0x07060504, 0x2110, v12, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v29;
//CHECK-NEXT:      int16_t * restrict r_v29_v6 = v6;
//CHECK-NEXT:      v29 = upd_w(v29, 0, *(v16int16 *)(r_v29_v6 + m2*v19+v22));
//CHECK-NEXT:      v25 = mac16(v25, v29, 0, 0x03020100, 0x07060504, 0x2110, v12, 8, 0, 0, 1);
//CHECK-NEXT:      v29 = upd_w(v29, 1, *(v16int16 *)(r_v29_v6 + m2*v19+v27 + 15));
//CHECK-NEXT:      v25 = mac16(v25, v29, 2, 0x03020100, 0x07060504, 0x2110, v12, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v30 = srs(v25, 0);
//CHECK-NEXT:      *(v16int16 *)(v8 + m5*v15+v22) = v30;
//CHECK-NEXT:    }
//CHECK-NEXT:  }


// CHECK-LABEL: void conv2d(int16_t * restrict v6, size_t m1, size_t m2, int16_t * restrict v7, size_t m3, int16_t * restrict v8, size_t m4, size_t m5) {
module  {
  func.func @conv2d(%arg0: memref<?x?xi16>, %arg1: memref<?xi16>, %arg2: memref<?x?xi16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xi16>
    %1 = memref.dim %arg0, %c1 : memref<?x?xi16>
    %2 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    scf.for %arg3 = %c0_0 to %0 step %c1_1 {
      %c1_2 = arith.constant 1 : index
      %3 = arith.addi %arg3, %c1_2 : index
      %c2 = arith.constant 2 : index
      %4 = arith.addi %arg3, %c2 : index
      %c0_3 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0_3 to %1 step %c16 {
        %5 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<16xi16>
        %6 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %7 = aievec.upd %arg0[%arg3, %arg4], %6 {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
        %8 = aievec.ups %5 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %9 = aievec.mac %7, %2, %8 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec.mac %7, %2, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %11 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %12 = aievec.upd %arg0[%3, %arg4], %11 {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
        %13 = aievec.mac %12, %2, %10 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec.mac %12, %2, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %15 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
        %16 = aievec.upd %arg0[%4, %arg4], %15 {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
        %17 = aievec.mac %16, %2, %14 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec.mac %16, %2, %17 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %19 = aievec.srs %18 {shift = 0 : i8} : vector<16xi48>, vector<16xi16>
        vector.transfer_write %19, %arg2[%arg3, %arg4] : vector<16xi16>, memref<?x?xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v9 = 0;
//CHECK-NEXT:  v16int16 v10 = *(v16int16 *)(v7 + v9);
//CHECK-NEXT:  size_t v11 = 0;
//CHECK-NEXT:  size_t v12 = 1;
//CHECK-NEXT:  for (size_t v13 = v11; v13 < m1; v13 += v12)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v14 = 1;
//CHECK-NEXT:    size_t v15 = v13 + v14;
//CHECK-NEXT:    size_t v16 = 2;
//CHECK-NEXT:    size_t v17 = v13 + v16;
//CHECK-NEXT:    size_t v18 = 0;
//CHECK-NEXT:    size_t v19 = 16;
//CHECK-NEXT:    for (size_t v20 = v18; v20 < m2; v20 += v19)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v21 = *(v16int16 *)(v8 + m5*v13+v20);
//CHECK-NEXT:      v32int16 v22;
//CHECK-NEXT:      int16_t * restrict r_v22_v6 = v6;
//CHECK-NEXT:      v22 = upd_w(v22, 0, *(v16int16 *)(r_v22_v6 + m2*v13+v20));
//CHECK-NEXT:      v22 = upd_w(v22, 1, *(v16int16 *)(r_v22_v6 + m2*v13+v20 + 16));
//CHECK-NEXT:      v16acc48 v23 = ups(v21, 0);
//CHECK-NEXT:      v23 = mac16(v23, v22, 0, 0x03020100, 0x07060504, 0x2110, v10, 0, 0, 0, 1);
//CHECK-NEXT:      v23 = mac16(v23, v22, 2, 0x03020100, 0x07060504, 0x2110, v10, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v24;
//CHECK-NEXT:      int16_t * restrict r_v24_v6 = v6;
//CHECK-NEXT:      v24 = upd_w(v24, 0, *(v16int16 *)(r_v24_v6 + m2*v15+v20));
//CHECK-NEXT:      v24 = upd_w(v24, 1, *(v16int16 *)(r_v24_v6 + m2*v15+v20 + 16));
//CHECK-NEXT:      v23 = mac16(v23, v24, 0, 0x03020100, 0x07060504, 0x2110, v10, 4, 0, 0, 1);
//CHECK-NEXT:      v23 = mac16(v23, v24, 2, 0x03020100, 0x07060504, 0x2110, v10, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v25;
//CHECK-NEXT:      int16_t * restrict r_v25_v6 = v6;
//CHECK-NEXT:      v25 = upd_w(v25, 0, *(v16int16 *)(r_v25_v6 + m2*v17+v20));
//CHECK-NEXT:      v25 = upd_w(v25, 1, *(v16int16 *)(r_v25_v6 + m2*v17+v20 + 16));
//CHECK-NEXT:      v23 = mac16(v23, v25, 0, 0x03020100, 0x07060504, 0x2110, v10, 8, 0, 0, 1);
//CHECK-NEXT:      v23 = mac16(v23, v25, 2, 0x03020100, 0x07060504, 0x2110, v10, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v26 = srs(v23, 0);
//CHECK-NEXT:      *(v16int16 *)(v8 + m5*v13+v20) = v26;
//CHECK-NEXT:    }
//CHECK-NEXT:  }

// CHECK-LABEL: void conv2d(int16_t * restrict v4, size_t m1, int16_t * restrict v5, size_t m2, int16_t * restrict v6, size_t m3) {
module  {
  func.func @conv2d(%arg0: memref<?x256xi16>, %arg1: memref<?xi16>, %arg2: memref<?x256xi16>) {
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x256xi16>
    %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0_0 to %0 step %c1 {
      %c1_1 = arith.constant 1 : index
      %2 = arith.addi %arg3, %c1_1 : index
      %c2 = arith.constant 2 : index
      %3 = arith.addi %arg3, %c2 : index
      %c0_2 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0_2 to %c256 step %c16 {
        %4 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<16xi16>
        %5 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
        %6 = aievec.upd %arg0[%arg3, %arg4], %5 {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
        %7 = aievec.ups %4 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %8 = aievec.mac %6, %1, %7 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %9 = aievec.mac %6, %1, %8 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %10 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
        %11 = aievec.upd %arg0[%2, %arg4], %10 {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
        %12 = aievec.mac %11, %1, %9 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %13 = aievec.mac %11, %1, %12 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %14 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
        %15 = aievec.upd %arg0[%3, %arg4], %14 {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
        %16 = aievec.mac %15, %1, %13 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %17 = aievec.mac %15, %1, %16 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %18 = aievec.srs %17 {shift = 0 : i8} : vector<16xi48>, vector<16xi16>
        vector.transfer_write %18, %arg2[%arg3, %arg4] : vector<16xi16>, memref<?x256xi16>
      }
    }
    return
  }
}

//CHECK-NEXT:  size_t v7 = 0;
//CHECK-NEXT:  v16int16 v8 = *(v16int16 *)(v5 + v7);
//CHECK-NEXT:  size_t v9 = 0;
//CHECK-NEXT:  size_t v10 = 1;
//CHECK-NEXT:  for (size_t v11 = v9; v11 < m1; v11 += v10)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v12 = 1;
//CHECK-NEXT:    size_t v13 = v11 + v12;
//CHECK-NEXT:    size_t v14 = 2;
//CHECK-NEXT:    size_t v15 = v11 + v14;
//CHECK-NEXT:    size_t v16 = 0;
//CHECK-NEXT:    size_t v17 = 256;
//CHECK-NEXT:    size_t v18 = 16;
//CHECK-NEXT:    for (size_t v19 = v16; v19 < v17; v19 += v18)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(16, 16)
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int16 v20 = *(v16int16 *)(v6 + 256*v11+v19);
//CHECK-NEXT:      v32int16 v21;
//CHECK-NEXT:      int16_t * restrict r_v21_v4 = v4;
//CHECK-NEXT:      v21 = upd_w(v21, 0, *(v16int16 *)(r_v21_v4 + 256*v11+v19));
//CHECK-NEXT:      v21 = upd_w(v21, 1, *(v16int16 *)(r_v21_v4 + 256*v11+v19 + 16));
//CHECK-NEXT:      v16acc48 v22 = ups(v20, 0);
//CHECK-NEXT:      v22 = mac16(v22, v21, 0, 0x03020100, 0x07060504, 0x2110, v8, 0, 0, 0, 1);
//CHECK-NEXT:      v22 = mac16(v22, v21, 2, 0x03020100, 0x07060504, 0x2110, v8, 2, 0, 0, 1);
//CHECK-NEXT:      v32int16 v23;
//CHECK-NEXT:      int16_t * restrict r_v23_v4 = v4;
//CHECK-NEXT:      v23 = upd_w(v23, 0, *(v16int16 *)(r_v23_v4 + 256*v13+v19));
//CHECK-NEXT:      v23 = upd_w(v23, 1, *(v16int16 *)(r_v23_v4 + 256*v13+v19 + 16));
//CHECK-NEXT:      v22 = mac16(v22, v23, 0, 0x03020100, 0x07060504, 0x2110, v8, 4, 0, 0, 1);
//CHECK-NEXT:      v22 = mac16(v22, v23, 2, 0x03020100, 0x07060504, 0x2110, v8, 6, 0, 0, 1);
//CHECK-NEXT:      v32int16 v24;
//CHECK-NEXT:      int16_t * restrict r_v24_v4 = v4;
//CHECK-NEXT:      v24 = upd_w(v24, 0, *(v16int16 *)(r_v24_v4 + 256*v15+v19));
//CHECK-NEXT:      v24 = upd_w(v24, 1, *(v16int16 *)(r_v24_v4 + 256*v15+v19 + 16));
//CHECK-NEXT:      v22 = mac16(v22, v24, 0, 0x03020100, 0x07060504, 0x2110, v8, 8, 0, 0, 1);
//CHECK-NEXT:      v22 = mac16(v22, v24, 2, 0x03020100, 0x07060504, 0x2110, v8, 10, 0, 0, 1);
//CHECK-NEXT:      v16int16 v25 = srs(v22, 0);
//CHECK-NEXT:      *(v16int16 *)(v6 + 256*v11+v19) = v25;
//CHECK-NEXT:    }
//CHECK-NEXT:  }

