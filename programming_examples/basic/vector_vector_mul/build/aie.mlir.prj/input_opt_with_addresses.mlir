module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @in1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @out_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @out_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @out_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @out() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in2_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in2() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in1_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @out_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @in2_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @in1_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @out_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @in2_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @in1_cons_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(52 : i32) : i32
    %13 = llvm.mlir.constant(51 : i32) : i32
    %14 = llvm.mlir.constant(49 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(16 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(2 : index) : i64
    %19 = llvm.mlir.constant(960 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb10
    %22 = llvm.icmp "slt" %21, %8 : i64
    llvm.cond_br %22, ^bb2(%20 : i64), ^bb11
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb9
    %24 = llvm.icmp "slt" %23, %19 : i64
    llvm.cond_br %24, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%25: i64):  // 2 preds: ^bb3, ^bb5
    %26 = llvm.icmp "slt" %25, %16 : i64
    llvm.cond_br %26, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %27 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.and %28, %5  : i64
    %30 = llvm.icmp "eq" %29, %20 : i64
    "llvm.intr.assume"(%30) : (i1) -> ()
    %31 = llvm.getelementptr %27[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %32 = llvm.load %31 : !llvm.ptr -> i32
    %33 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %5  : i64
    %36 = llvm.icmp "eq" %35, %20 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.getelementptr %33[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %38 = llvm.load %37 : !llvm.ptr -> i32
    %39 = llvm.mul %32, %38 : i32
    %40 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.and %41, %5  : i64
    %43 = llvm.icmp "eq" %42, %20 : i64
    "llvm.intr.assume"(%43) : (i1) -> ()
    %44 = llvm.getelementptr %40[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %39, %44 : i32, !llvm.ptr
    %45 = llvm.add %25, %7 : i64
    llvm.br ^bb4(%45 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%46: i64):  // 2 preds: ^bb6, ^bb8
    %47 = llvm.icmp "slt" %46, %16 : i64
    llvm.cond_br %47, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %48 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    %52 = llvm.getelementptr %48[%46] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %53 = llvm.load %52 : !llvm.ptr -> i32
    %54 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %54[%46] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %59 = llvm.load %58 : !llvm.ptr -> i32
    %60 = llvm.mul %53, %59 : i32
    %61 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.and %62, %5  : i64
    %64 = llvm.icmp "eq" %63, %20 : i64
    "llvm.intr.assume"(%64) : (i1) -> ()
    %65 = llvm.getelementptr %61[%46] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %60, %65 : i32, !llvm.ptr
    %66 = llvm.add %46, %7 : i64
    llvm.br ^bb7(%66 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %67 = llvm.add %23, %18 : i64
    llvm.br ^bb2(%67 : i64)
  ^bb10:  // pred: ^bb2
    %68 = llvm.add %21, %7 : i64
    llvm.br ^bb1(%68 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
}

