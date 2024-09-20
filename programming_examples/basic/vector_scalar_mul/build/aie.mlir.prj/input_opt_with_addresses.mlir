module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @in_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @infactor_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external @infactor_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external @out_buff_1() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @out_buff_0() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @out_cons() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @out() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @infactor_cons() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external @infactor() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external @in_cons() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.mlir.global external @in() {addr_space = 0 : i32} : !llvm.array<3072 x i32>
  llvm.func @vector_scalar_mul_int32_scalar(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @vector_scalar_mul_int32_vector(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @infactor_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @in_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @out_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @in_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @infactor_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @out_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(9223372036854775806 : index) : i64
    %9 = llvm.mlir.constant(50 : i32) : i32
    %10 = llvm.mlir.constant(53 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(52 : i32) : i32
    %14 = llvm.mlir.constant(51 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(3072 : i32) : i32
    %17 = llvm.mlir.constant(4 : index) : i64
    %18 = llvm.mlir.constant(-1 : i32) : i32
    %19 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%19 : i64)
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb8
    %21 = llvm.icmp "slt" %20, %8 : i64
    llvm.cond_br %21, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %18) : (i32, i32) -> ()
    llvm.br ^bb3(%19 : i64)
  ^bb3(%22: i64):  // 2 preds: ^bb2, ^bb4
    %23 = llvm.icmp "slt" %22, %17 : i64
    llvm.cond_br %23, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %24 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.and %25, %5  : i64
    %27 = llvm.icmp "eq" %26, %19 : i64
    "llvm.intr.assume"(%27) : (i1) -> ()
    %28 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.and %29, %5  : i64
    %31 = llvm.icmp "eq" %30, %19 : i64
    "llvm.intr.assume"(%31) : (i1) -> ()
    %32 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.and %33, %5  : i64
    %35 = llvm.icmp "eq" %34, %19 : i64
    "llvm.intr.assume"(%35) : (i1) -> ()
    llvm.call @vector_scalar_mul_int32_vector(%32, %24, %28, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %36 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.and %37, %5  : i64
    %39 = llvm.icmp "eq" %38, %19 : i64
    "llvm.intr.assume"(%39) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    %40 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.and %41, %5  : i64
    %43 = llvm.icmp "eq" %42, %19 : i64
    "llvm.intr.assume"(%43) : (i1) -> ()
    llvm.call @vector_scalar_mul_int32_vector(%40, %36, %28, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %44 = llvm.add %22, %7 : i64
    llvm.br ^bb3(%44 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %18) : (i32, i32) -> ()
    llvm.br ^bb6(%19 : i64)
  ^bb6(%45: i64):  // 2 preds: ^bb5, ^bb7
    %46 = llvm.icmp "slt" %45, %17 : i64
    llvm.cond_br %46, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %47 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.and %48, %5  : i64
    %50 = llvm.icmp "eq" %49, %19 : i64
    "llvm.intr.assume"(%50) : (i1) -> ()
    %51 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.and %52, %5  : i64
    %54 = llvm.icmp "eq" %53, %19 : i64
    "llvm.intr.assume"(%54) : (i1) -> ()
    %55 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %5  : i64
    %58 = llvm.icmp "eq" %57, %19 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.call @vector_scalar_mul_int32_vector(%55, %47, %51, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %59 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %60 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %61 = llvm.and %60, %5  : i64
    %62 = llvm.icmp "eq" %61, %19 : i64
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    %63 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.and %64, %5  : i64
    %66 = llvm.icmp "eq" %65, %19 : i64
    "llvm.intr.assume"(%66) : (i1) -> ()
    llvm.call @vector_scalar_mul_int32_vector(%63, %59, %51, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %67 = llvm.add %45, %7 : i64
    llvm.br ^bb6(%67 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %68 = llvm.add %20, %7 : i64
    llvm.br ^bb1(%68 : i64)
  ^bb9:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %18) : (i32, i32) -> ()
    llvm.br ^bb10(%19 : i64)
  ^bb10(%69: i64):  // 2 preds: ^bb9, ^bb11
    %70 = llvm.icmp "slt" %69, %17 : i64
    llvm.cond_br %70, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %71 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %73 = llvm.and %72, %5  : i64
    %74 = llvm.icmp "eq" %73, %19 : i64
    "llvm.intr.assume"(%74) : (i1) -> ()
    %75 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    %76 = llvm.ptrtoint %75 : !llvm.ptr to i64
    %77 = llvm.and %76, %5  : i64
    %78 = llvm.icmp "eq" %77, %19 : i64
    "llvm.intr.assume"(%78) : (i1) -> ()
    %79 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %80 = llvm.ptrtoint %79 : !llvm.ptr to i64
    %81 = llvm.and %80, %5  : i64
    %82 = llvm.icmp "eq" %81, %19 : i64
    "llvm.intr.assume"(%82) : (i1) -> ()
    llvm.call @vector_scalar_mul_int32_vector(%79, %71, %75, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %83 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.and %84, %5  : i64
    %86 = llvm.icmp "eq" %85, %19 : i64
    "llvm.intr.assume"(%86) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    %87 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3072 x i32>
    %88 = llvm.ptrtoint %87 : !llvm.ptr to i64
    %89 = llvm.and %88, %5  : i64
    %90 = llvm.icmp "eq" %89, %19 : i64
    "llvm.intr.assume"(%90) : (i1) -> ()
    llvm.call @vector_scalar_mul_int32_vector(%87, %83, %75, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %91 = llvm.add %69, %7 : i64
    llvm.br ^bb10(%91 : i64)
  ^bb12:  // pred: ^bb10
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.return
  }
}

