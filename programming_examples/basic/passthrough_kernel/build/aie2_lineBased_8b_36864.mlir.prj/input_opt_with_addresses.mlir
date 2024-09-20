module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @in_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.mlir.global external @in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.mlir.global external @out_buff_1() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.mlir.global external @out_buff_0() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @out_cons() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.mlir.global external @out() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.mlir.global external @in_cons() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.mlir.global external @in() {addr_space = 0 : i32} : !llvm.array<9216 x i8>
  llvm.func @passThroughLine(!llvm.ptr, !llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @in_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @out_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @in_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(31 : index) : i64
    %4 = llvm.mlir.addressof @out_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(9223372036854775806 : index) : i64
    %7 = llvm.mlir.constant(51 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(49 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(9216 : i32) : i32
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb2
    %16 = llvm.icmp "slt" %15, %6 : i64
    llvm.cond_br %16, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    %17 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<9216 x i8>
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.and %18, %3  : i64
    %20 = llvm.icmp "eq" %19, %14 : i64
    "llvm.intr.assume"(%20) : (i1) -> ()
    %21 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<9216 x i8>
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.and %22, %3  : i64
    %24 = llvm.icmp "eq" %23, %14 : i64
    "llvm.intr.assume"(%24) : (i1) -> ()
    llvm.call @passThroughLine(%21, %17, %12) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    %25 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<9216 x i8>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %3  : i64
    %28 = llvm.icmp "eq" %27, %14 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    %29 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<9216 x i8>
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.and %30, %3  : i64
    %32 = llvm.icmp "eq" %31, %14 : i64
    "llvm.intr.assume"(%32) : (i1) -> ()
    llvm.call @passThroughLine(%29, %25, %12) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    %33 = llvm.add %15, %5 : i64
    llvm.br ^bb1(%33 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    %34 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<9216 x i8>
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.and %35, %3  : i64
    %37 = llvm.icmp "eq" %36, %14 : i64
    "llvm.intr.assume"(%37) : (i1) -> ()
    %38 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<9216 x i8>
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.and %39, %3  : i64
    %41 = llvm.icmp "eq" %40, %14 : i64
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @passThroughLine(%38, %34, %12) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    llvm.return
  }
}

