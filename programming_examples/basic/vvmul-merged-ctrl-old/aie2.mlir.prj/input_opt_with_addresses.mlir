module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @objFifo_out0_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_out0_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_out1_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_out1_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @objFifo_in0() {addr_space = 0 : i32} : !llvm.array<56 x array<56 x i8>>
  llvm.mlir.global external @objFifo_out0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @objFifo_out1_buff_0 : !llvm.ptr
    %1 = llvm.mlir.constant(31 : index) : i64
    %2 = llvm.mlir.addressof @objFifo_in1_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(49 : i32) : i32
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.constant(12 : i8) : i8
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(51 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(50 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(-1 : i32) : i32
    %12 = llvm.mlir.constant(0 : index) : i64
    llvm.call @llvm.aie2.acquire(%3, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %11) : (i32, i32) -> ()
    llvm.br ^bb1(%12 : i64)
  ^bb1(%13: i64):  // 2 preds: ^bb0, ^bb4
    %14 = llvm.icmp "slt" %13, %4 : i64
    llvm.cond_br %14, ^bb2(%12 : i64), ^bb5
  ^bb2(%15: i64):  // 2 preds: ^bb1, ^bb3
    %16 = llvm.icmp "slt" %15, %4 : i64
    llvm.cond_br %16, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %17 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x i8>>
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.and %18, %1  : i64
    %20 = llvm.icmp "eq" %19, %12 : i64
    "llvm.intr.assume"(%20) : (i1) -> ()
    %21 = llvm.mul %13, %4 : i64
    %22 = llvm.add %21, %15 : i64
    %23 = llvm.getelementptr %17[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %24 = llvm.load %23 : !llvm.ptr -> i8
    %25 = llvm.add %24, %5 : i8
    %26 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x i8>>
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %28 = llvm.and %27, %1  : i64
    %29 = llvm.icmp "eq" %28, %12 : i64
    "llvm.intr.assume"(%29) : (i1) -> ()
    %30 = llvm.getelementptr %26[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %25, %30 : i8, !llvm.ptr
    %31 = llvm.add %15, %6 : i64
    llvm.br ^bb2(%31 : i64)
  ^bb4:  // pred: ^bb2
    %32 = llvm.add %13, %6 : i64
    llvm.br ^bb1(%32 : i64)
  ^bb5:  // pred: ^bb1
    llvm.call @llvm.aie2.release(%8, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.return
  }
}

