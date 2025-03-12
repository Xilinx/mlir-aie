module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @inA_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.mlir.global external @inA_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.mlir.global external @memA_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.mlir.global external @memA_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.mlir.global external @inB_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @inB_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @memB_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @memB_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @memC_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.mlir.global external @memC_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.mlir.global external @memC_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.mlir.global external @memC_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @outC_cons() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.mlir.global external @outC() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.mlir.global external @memC_cons() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.mlir.global external @memC() {addr_space = 0 : i32} : !llvm.array<64 x array<96 x i16>>
  llvm.mlir.global external @memB_cons() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @memB() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @inB_cons() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @inB() {addr_space = 0 : i32} : !llvm.array<120 x array<96 x i8>>
  llvm.mlir.global external @memA_cons() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.mlir.global external @memA() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.mlir.global external @inA_cons() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.mlir.global external @inA() {addr_space = 0 : i32} : !llvm.array<64 x array<120 x i8>>
  llvm.func @zero_i16(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @matmul_i8_i16(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @memC_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @memA_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @memB_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @memA_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @memB_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(32 : index) : i64
    %6 = llvm.mlir.constant(true) : i1
    %7 = llvm.mlir.addressof @memC_buff_0 : !llvm.ptr
    %8 = llvm.mlir.constant(53 : i32) : i32
    %9 = llvm.mlir.constant(50 : i32) : i32
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(51 : i32) : i32
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(52 : i32) : i32
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.mlir.constant(4 : index) : i64
    %16 = llvm.mlir.constant(-1 : i32) : i32
    %17 = llvm.mlir.constant(2 : index) : i64
    %18 = llvm.mlir.constant(40 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.mlir.constant(4294967295 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%19 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb10
    %23 = llvm.icmp "slt" %22, %20 : i64
    llvm.cond_br %23, ^bb2(%19 : i64), ^bb11
  ^bb2(%24: i64):  // 2 preds: ^bb1, ^bb9
    %25 = llvm.icmp "slt" %24, %18 : i64
    llvm.cond_br %25, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    %26 = llvm.getelementptr %7[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<96 x i16>>
    llvm.intr.assume %6 ["align"(%26, %5 : !llvm.ptr, i64)] : i1
    llvm.call @zero_i16(%26) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%19 : i64)
  ^bb4(%27: i64):  // 2 preds: ^bb3, ^bb5
    %28 = llvm.icmp "slt" %27, %15 : i64
    llvm.cond_br %28, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%12, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %16) : (i32, i32) -> ()
    llvm.intr.assume %6 ["align"(%26, %5 : !llvm.ptr, i64)] : i1
    %29 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<120 x array<96 x i8>>
    llvm.intr.assume %6 ["align"(%29, %5 : !llvm.ptr, i64)] : i1
    %30 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<120 x i8>>
    llvm.intr.assume %6 ["align"(%30, %5 : !llvm.ptr, i64)] : i1
    llvm.call @matmul_i8_i16(%30, %29, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %16) : (i32, i32) -> ()
    llvm.intr.assume %6 ["align"(%26, %5 : !llvm.ptr, i64)] : i1
    %31 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<120 x array<96 x i8>>
    llvm.intr.assume %6 ["align"(%31, %5 : !llvm.ptr, i64)] : i1
    %32 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<120 x i8>>
    llvm.intr.assume %6 ["align"(%32, %5 : !llvm.ptr, i64)] : i1
    llvm.call @matmul_i8_i16(%32, %31, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    %33 = llvm.add %27, %17 : i64
    llvm.br ^bb4(%33 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%8, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    %34 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<96 x i16>>
    llvm.intr.assume %6 ["align"(%34, %5 : !llvm.ptr, i64)] : i1
    llvm.call @zero_i16(%34) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%19 : i64)
  ^bb7(%35: i64):  // 2 preds: ^bb6, ^bb8
    %36 = llvm.icmp "slt" %35, %15 : i64
    llvm.cond_br %36, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%12, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %16) : (i32, i32) -> ()
    llvm.intr.assume %6 ["align"(%34, %5 : !llvm.ptr, i64)] : i1
    %37 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<120 x array<96 x i8>>
    llvm.intr.assume %6 ["align"(%37, %5 : !llvm.ptr, i64)] : i1
    %38 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<120 x i8>>
    llvm.intr.assume %6 ["align"(%38, %5 : !llvm.ptr, i64)] : i1
    llvm.call @matmul_i8_i16(%38, %37, %34) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %16) : (i32, i32) -> ()
    llvm.intr.assume %6 ["align"(%34, %5 : !llvm.ptr, i64)] : i1
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<120 x array<96 x i8>>
    llvm.intr.assume %6 ["align"(%39, %5 : !llvm.ptr, i64)] : i1
    %40 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<120 x i8>>
    llvm.intr.assume %6 ["align"(%40, %5 : !llvm.ptr, i64)] : i1
    llvm.call @matmul_i8_i16(%40, %39, %34) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    %41 = llvm.add %35, %17 : i64
    llvm.br ^bb7(%41 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%8, %14) : (i32, i32) -> ()
    %42 = llvm.add %24, %17 : i64
    llvm.br ^bb2(%42 : i64)
  ^bb10:  // pred: ^bb2
    %43 = llvm.add %22, %21 : i64
    llvm.br ^bb1(%43 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
}

