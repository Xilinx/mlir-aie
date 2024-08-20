module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @objFifo_in0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @objFifo_in0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @objFifo_in1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_in1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_out1_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_out1_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_out0_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @objFifo_out0_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @objFifo_out0_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @objFifo_out0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @objFifo_out1_cons() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_out1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_in1_cons() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_in1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_in0_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @objFifo_in0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @objFifo_out1_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @objFifo_in1_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @objFifo_out1_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(31 : index) : i64
    %4 = llvm.mlir.addressof @objFifo_in1_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(51 : i32) : i32
    %9 = llvm.mlir.constant(48 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(49 : i32) : i32
    %12 = llvm.mlir.constant(-1 : i32) : i32
    %13 = llvm.mlir.constant(8 : index) : i64
    %14 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb8
    %16 = llvm.icmp "slt" %15, %13 : i64
    llvm.cond_br %16, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%11, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %12) : (i32, i32) -> ()
    llvm.br ^bb3(%14 : i64)
  ^bb3(%17: i64):  // 2 preds: ^bb2, ^bb4
    %18 = llvm.icmp "slt" %17, %13 : i64
    llvm.cond_br %18, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %19 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.and %20, %3  : i64
    %22 = llvm.icmp "eq" %21, %14 : i64
    "llvm.intr.assume"(%22) : (i1) -> ()
    %23 = llvm.getelementptr %19[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %24 = llvm.load %23 : !llvm.ptr -> i32
    %25 = llvm.add %24, %6 : i32
    %26 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %28 = llvm.and %27, %3  : i64
    %29 = llvm.icmp "eq" %28, %14 : i64
    "llvm.intr.assume"(%29) : (i1) -> ()
    %30 = llvm.getelementptr %26[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %25, %30 : i32, !llvm.ptr
    %31 = llvm.add %17, %7 : i64
    llvm.br ^bb3(%31 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%9, %6) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %6) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %12) : (i32, i32) -> ()
    llvm.br ^bb6(%14 : i64)
  ^bb6(%32: i64):  // 2 preds: ^bb5, ^bb7
    %33 = llvm.icmp "slt" %32, %13 : i64
    llvm.cond_br %33, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %34 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.and %35, %3  : i64
    %37 = llvm.icmp "eq" %36, %14 : i64
    "llvm.intr.assume"(%37) : (i1) -> ()
    %38 = llvm.getelementptr %34[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %39 = llvm.load %38 : !llvm.ptr -> i32
    %40 = llvm.add %39, %6 : i32
    %41 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %3  : i64
    %44 = llvm.icmp "eq" %43, %14 : i64
    "llvm.intr.assume"(%44) : (i1) -> ()
    %45 = llvm.getelementptr %41[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %40, %45 : i32, !llvm.ptr
    %46 = llvm.add %32, %7 : i64
    llvm.br ^bb6(%46 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.release(%9, %6) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %6) : (i32, i32) -> ()
    %47 = llvm.add %15, %5 : i64
    llvm.br ^bb1(%47 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}

