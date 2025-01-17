module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @shim_to_mem_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @shim_to_mem_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @mem_to_comp_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @mem_to_comp_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @comp_to_mem_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @comp_to_mem_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @comp_to_mem_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @comp_to_mem_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @mem_to_shim_cons() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @mem_to_shim() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @comp_to_mem_cons() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @comp_to_mem() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @mem_to_comp_cons() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @mem_to_comp() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @shim_to_mem_cons() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.mlir.global external @shim_to_mem() {addr_space = 0 : i32} : !llvm.array<8 x array<20 x i32>>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @comp_to_mem_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @mem_to_comp_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @comp_to_mem_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.addressof @mem_to_comp_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(51 : i32) : i32
    %7 = llvm.mlir.constant(48 : i32) : i32
    %8 = llvm.mlir.constant(50 : i32) : i32
    %9 = llvm.mlir.constant(49 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(20 : index) : i64
    %12 = llvm.mlir.constant(8 : index) : i64
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(9223372036854775806 : index) : i64
    %17 = llvm.mlir.constant(2 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb12
    %19 = llvm.icmp "slt" %18, %16 : i64
    llvm.cond_br %19, ^bb2, ^bb13
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    llvm.br ^bb3(%14 : i64)
  ^bb3(%20: i64):  // 2 preds: ^bb2, ^bb6
    %21 = llvm.icmp "slt" %20, %12 : i64
    llvm.cond_br %21, ^bb4(%14 : i64), ^bb7
  ^bb4(%22: i64):  // 2 preds: ^bb3, ^bb5
    %23 = llvm.icmp "slt" %22, %11 : i64
    llvm.cond_br %23, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %24 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<20 x i32>>
    llvm.intr.assume %4 ["align"(%24, %3 : !llvm.ptr, i64)] : i1
    %25 = llvm.mul %20, %11 : i64
    %26 = llvm.add %25, %22 : i64
    %27 = llvm.getelementptr %24[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %28 = llvm.load %27 : !llvm.ptr -> i32
    %29 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<20 x i32>>
    llvm.intr.assume %4 ["align"(%29, %3 : !llvm.ptr, i64)] : i1
    %30 = llvm.getelementptr %29[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %28, %30 : i32, !llvm.ptr
    %31 = llvm.add %22, %15 : i64
    llvm.br ^bb4(%31 : i64)
  ^bb6:  // pred: ^bb4
    %32 = llvm.add %20, %15 : i64
    llvm.br ^bb3(%32 : i64)
  ^bb7:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    llvm.br ^bb8(%14 : i64)
  ^bb8(%33: i64):  // 2 preds: ^bb7, ^bb11
    %34 = llvm.icmp "slt" %33, %12 : i64
    llvm.cond_br %34, ^bb9(%14 : i64), ^bb12
  ^bb9(%35: i64):  // 2 preds: ^bb8, ^bb10
    %36 = llvm.icmp "slt" %35, %11 : i64
    llvm.cond_br %36, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %37 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<20 x i32>>
    llvm.intr.assume %4 ["align"(%37, %3 : !llvm.ptr, i64)] : i1
    %38 = llvm.mul %33, %11 : i64
    %39 = llvm.add %38, %35 : i64
    %40 = llvm.getelementptr %37[%39] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %41 = llvm.load %40 : !llvm.ptr -> i32
    %42 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<20 x i32>>
    llvm.intr.assume %4 ["align"(%42, %3 : !llvm.ptr, i64)] : i1
    %43 = llvm.getelementptr %42[%39] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %41, %43 : i32, !llvm.ptr
    %44 = llvm.add %35, %15 : i64
    llvm.br ^bb9(%44 : i64)
  ^bb11:  // pred: ^bb9
    %45 = llvm.add %33, %15 : i64
    llvm.br ^bb8(%45 : i64)
  ^bb12:  // pred: ^bb8
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    %46 = llvm.add %18, %17 : i64
    llvm.br ^bb1(%46 : i64)
  ^bb13:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    llvm.br ^bb14(%14 : i64)
  ^bb14(%47: i64):  // 2 preds: ^bb13, ^bb17
    %48 = llvm.icmp "slt" %47, %12 : i64
    llvm.cond_br %48, ^bb15(%14 : i64), ^bb18
  ^bb15(%49: i64):  // 2 preds: ^bb14, ^bb16
    %50 = llvm.icmp "slt" %49, %11 : i64
    llvm.cond_br %50, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %51 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<20 x i32>>
    llvm.intr.assume %4 ["align"(%51, %3 : !llvm.ptr, i64)] : i1
    %52 = llvm.mul %47, %11 : i64
    %53 = llvm.add %52, %49 : i64
    %54 = llvm.getelementptr %51[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %55 = llvm.load %54 : !llvm.ptr -> i32
    %56 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<20 x i32>>
    llvm.intr.assume %4 ["align"(%56, %3 : !llvm.ptr, i64)] : i1
    %57 = llvm.getelementptr %56[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %55, %57 : i32, !llvm.ptr
    %58 = llvm.add %49, %15 : i64
    llvm.br ^bb15(%58 : i64)
  ^bb17:  // pred: ^bb15
    %59 = llvm.add %47, %15 : i64
    llvm.br ^bb14(%59 : i64)
  ^bb18:  // pred: ^bb14
    llvm.call @llvm.aie2.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %10) : (i32, i32) -> ()
    llvm.return
  }
}

