module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @A_L2L1_0_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_0_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_0_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_0_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_1_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_1_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_1_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_1_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_2_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_2_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_2_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_2_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_3_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_3_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_3_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_3_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L3L2_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @A_L3L2_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @A_L3L2_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @A_L3L2_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @B_L3L2_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @B_L3L2_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @B_L2L1_0_3_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_3_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L3L2_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @B_L3L2_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @B_L2L1_1_3_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_3_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @C_L1L2_0_0_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_0_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_1_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_1_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_2_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_2_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_3_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_3_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L2L3_0_buff_1() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @C_L2L3_0_buff_0() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @C_L1L2_1_0_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_0_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_1_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_1_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_3_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_3_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L2L3_1_buff_1() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @C_L2L3_1_buff_0() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @C_L2L3_1_cons() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @C_L2L3_1() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @C_L1L2_1_3_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_3() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_2_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_2() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_1_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_0_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_1_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L2L3_0_cons() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @C_L2L3_0() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @C_L1L2_0_3_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_3() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_2_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_2() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_1_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_1() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_0_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @C_L1L2_0_0() {addr_space = 0 : i32} : !llvm.array<16 x array<48 x i32>>
  llvm.mlir.global external @B_L2L1_1_0_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_1_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_2_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1_3_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_1() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L3L2_1_cons() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @B_L3L2_1() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @B_L2L1_0_0_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_1_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_2_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0_3_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L2L1_0() {addr_space = 0 : i32} : !llvm.array<32 x array<48 x i16>>
  llvm.mlir.global external @B_L3L2_0_cons() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @B_L3L2_0() {addr_space = 0 : i32} : !llvm.array<1536 x i16>
  llvm.mlir.global external @A_L3L2_1_cons() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @A_L3L2_1() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @A_L3L2_0_cons() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @A_L3L2_0() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @A_L2L1_3_0_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_3_1_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_3() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_2_0_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_2_1_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_2() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_1_0_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_1_1_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_1() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_0_0_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_0_1_cons() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.mlir.global external @A_L2L1_0() {addr_space = 0 : i32} : !llvm.array<16 x array<32 x i16>>
  llvm.func @zero_scalar_i32(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @zero_i32(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @matmul_scalar_i16_i32(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @matmul_i16_i32(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_1_5() {
    %0 = llvm.mlir.addressof @C_L1L2_1_3_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_3_1_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_1_3_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_3_1_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_1_3_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_1_3_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(4294967295 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(80 : index) : i64
    %19 = llvm.mlir.constant(2 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb10
    %22 = llvm.icmp "slt" %21, %7 : i64
    llvm.cond_br %22, ^bb2(%20 : i64), ^bb11
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb9
    %24 = llvm.icmp "slt" %23, %18 : i64
    llvm.cond_br %24, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %19 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %19 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %19 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %8 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_5() {
    %0 = llvm.mlir.addressof @C_L1L2_0_3_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_3_0_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_0_3_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_3_0_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_0_3_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_0_3_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(4294967295 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(80 : index) : i64
    %19 = llvm.mlir.constant(2 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb10
    %22 = llvm.icmp "slt" %21, %7 : i64
    llvm.cond_br %22, ^bb2(%20 : i64), ^bb11
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb9
    %24 = llvm.icmp "slt" %23, %18 : i64
    llvm.cond_br %24, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %19 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %19 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %19 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %8 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_1_4() {
    %0 = llvm.mlir.addressof @C_L1L2_1_2_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_2_1_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_1_2_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_2_1_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_1_2_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_1_2_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(4294967295 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(80 : index) : i64
    %19 = llvm.mlir.constant(2 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb10
    %22 = llvm.icmp "slt" %21, %7 : i64
    llvm.cond_br %22, ^bb2(%20 : i64), ^bb11
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb9
    %24 = llvm.icmp "slt" %23, %18 : i64
    llvm.cond_br %24, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %19 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %19 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %19 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %8 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_4() {
    %0 = llvm.mlir.addressof @C_L1L2_0_2_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_2_0_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_0_2_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_2_0_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_0_2_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_0_2_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(4294967295 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(80 : index) : i64
    %19 = llvm.mlir.constant(2 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb10
    %22 = llvm.icmp "slt" %21, %7 : i64
    llvm.cond_br %22, ^bb2(%20 : i64), ^bb11
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb9
    %24 = llvm.icmp "slt" %23, %18 : i64
    llvm.cond_br %24, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %19 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %19 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %19 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %8 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_1_3() {
    %0 = llvm.mlir.addressof @C_L1L2_1_1_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_1_1_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_1_1_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_1_1_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_1_1_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_1_1_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(4294967295 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(80 : index) : i64
    %19 = llvm.mlir.constant(2 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb10
    %22 = llvm.icmp "slt" %21, %7 : i64
    llvm.cond_br %22, ^bb2(%20 : i64), ^bb11
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb9
    %24 = llvm.icmp "slt" %23, %18 : i64
    llvm.cond_br %24, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %19 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %19 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %19 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %8 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_3() {
    %0 = llvm.mlir.addressof @C_L1L2_0_1_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_1_0_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_0_1_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_1_0_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_0_1_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_0_1_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(4294967295 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(80 : index) : i64
    %19 = llvm.mlir.constant(2 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb10
    %22 = llvm.icmp "slt" %21, %7 : i64
    llvm.cond_br %22, ^bb2(%20 : i64), ^bb11
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb9
    %24 = llvm.icmp "slt" %23, %18 : i64
    llvm.cond_br %24, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %19 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %19 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %19 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %8 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_1_2() {
    %0 = llvm.mlir.addressof @C_L1L2_1_0_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_0_1_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_1_0_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_0_1_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_1_0_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_1_0_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(4294967295 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(2 : index) : i64
    %19 = llvm.mlir.constant(80 : index) : i64
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
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %18 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %18 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %18 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %7 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @C_L1L2_0_0_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L2L1_0_0_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L2L1_0_0_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @A_L2L1_0_0_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_L2L1_0_0_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @C_L1L2_0_0_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(4294967295 : index) : i64
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(52 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(2 : index) : i64
    %19 = llvm.mlir.constant(80 : index) : i64
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
    %25 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.and %26, %5  : i64
    %28 = llvm.icmp "eq" %27, %20 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @zero_i32(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%20 : i64)
  ^bb4(%29: i64):  // 2 preds: ^bb3, ^bb5
    %30 = llvm.icmp "slt" %29, %16 : i64
    llvm.cond_br %30, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %31 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %5  : i64
    %34 = llvm.icmp "eq" %33, %20 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %5  : i64
    %38 = llvm.icmp "eq" %37, %20 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @matmul_i16_i32(%35, %31, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%28) : (i1) -> ()
    %39 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %5  : i64
    %42 = llvm.icmp "eq" %41, %20 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %5  : i64
    %46 = llvm.icmp "eq" %45, %20 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @matmul_i16_i32(%43, %39, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %47 = llvm.add %29, %18 : i64
    llvm.br ^bb4(%47 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<48 x i32>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @zero_i32(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%20 : i64)
  ^bb7(%52: i64):  // 2 preds: ^bb6, ^bb8
    %53 = llvm.icmp "slt" %52, %16 : i64
    llvm.cond_br %53, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %54 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %5  : i64
    %57 = llvm.icmp "eq" %56, %20 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    %58 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %20 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @matmul_i16_i32(%58, %54, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    %62 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<48 x i16>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %5  : i64
    %65 = llvm.icmp "eq" %64, %20 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    %66 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x array<32 x i16>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %5  : i64
    %69 = llvm.icmp "eq" %68, %20 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    llvm.call @matmul_i16_i32(%66, %62, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %70 = llvm.add %52, %18 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %71 = llvm.add %23, %18 : i64
    llvm.br ^bb2(%71 : i64)
  ^bb10:  // pred: ^bb2
    %72 = llvm.add %21, %7 : i64
    llvm.br ^bb1(%72 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
}

