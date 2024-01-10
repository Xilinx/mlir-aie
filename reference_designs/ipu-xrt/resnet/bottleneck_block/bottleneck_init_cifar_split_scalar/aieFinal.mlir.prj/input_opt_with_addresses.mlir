module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @rtp5() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp4() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp3() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp2() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @inOF_act_L3L2_1_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_1_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @skip_buf_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @skip_buf_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_wts_0_L3L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<73728 x i8>
  llvm.mlir.global external @wts_buf_00_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.mlir.global external @wts_buf_01_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<36864 x i8>
  llvm.mlir.global external @wts_buf_01_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<36864 x i8>
  llvm.mlir.global external @wts_buf_02_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32768 x i8>
  llvm.mlir.global external @act_2_3_4_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_1_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_1_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_0_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_0_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_3_5_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<32 x i8>>>
  llvm.mlir.global external @act_3_5_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<32 x i8>>>
  llvm.mlir.global external @act_4_5_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<32 x i8>>>
  llvm.mlir.global external @act_4_5_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<32 x i8>>>
  llvm.mlir.global external @outOFL2L3_buff_1() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @outOFL2L3_buff_0() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<256 x i8>>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @outOFL2L3_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @outOFL2L3() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @act_4_5() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<32 x i8>>>
  llvm.mlir.global external @act_3_5() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<32 x i8>>>
  llvm.mlir.global external @act_2_3_4_0_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4_1_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_2_3_4() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @wts_buf_02_cons() {addr_space = 0 : i32} : !llvm.array<32768 x i8>
  llvm.mlir.global external @wts_buf_02() {addr_space = 0 : i32} : !llvm.array<32768 x i8>
  llvm.mlir.global external @wts_buf_01_0_cons() {addr_space = 0 : i32} : !llvm.array<36864 x i8>
  llvm.mlir.global external @wts_buf_01_1_cons() {addr_space = 0 : i32} : !llvm.array<36864 x i8>
  llvm.mlir.global external @wts_buf_01() {addr_space = 0 : i32} : !llvm.array<36864 x i8>
  llvm.mlir.global external @wts_buf_00_cons() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.mlir.global external @wts_buf_00() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.mlir.global external @inOF_wts_0_L3L2_cons() {addr_space = 0 : i32} : !llvm.array<73728 x i8>
  llvm.mlir.global external @inOF_wts_0_L3L2() {addr_space = 0 : i32} : !llvm.array<73728 x i8>
  llvm.mlir.global external @skip_buf_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @skip_buf() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_0_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_1_cons() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2() {addr_space = 0 : i32} : !llvm.array<32 x array<1 x array<64 x i8>>>
  llvm.func @conv2dk1(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @conv2dk3(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @conv2dk1_skip(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @sequence(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.return
  }
  llvm.func @core_0_4() {
    %0 = llvm.mlir.constant(31 : index) : i64
    %1 = llvm.mlir.constant(4294967295 : index) : i64
    %2 = llvm.mlir.constant(256 : i32) : i32
    %3 = llvm.mlir.constant(64 : i32) : i32
    %4 = llvm.mlir.constant(32 : i32) : i32
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(50 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(36 : i32) : i32
    %10 = llvm.mlir.constant(4 : i32) : i32
    %11 = llvm.mlir.constant(53 : i32) : i32
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(52 : i32) : i32
    %14 = llvm.mlir.constant(37 : i32) : i32
    %15 = llvm.mlir.constant(5 : i32) : i32
    %16 = llvm.mlir.constant(51 : i32) : i32
    %17 = llvm.mlir.constant(1 : i32) : i32
    %18 = llvm.mlir.constant(-1 : i32) : i32
    %19 = llvm.mlir.constant(32 : index) : i64
    %20 = llvm.mlir.constant(2 : index) : i64
    llvm.br ^bb1(%6 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb5
    %22 = llvm.icmp "slt" %21, %1 : i64
    llvm.cond_br %22, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%16, %18) : (i32, i32) -> ()
    %23 = llvm.mlir.addressof @rtp5 : !llvm.ptr
    %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.and %25, %0  : i64
    %27 = llvm.icmp "eq" %26, %6 : i64
    "llvm.intr.assume"(%27) : (i1) -> ()
    %28 = llvm.load %24 : !llvm.ptr -> i32
    "llvm.intr.assume"(%27) : (i1) -> ()
    %29 = llvm.getelementptr %24[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %30 = llvm.load %29 : !llvm.ptr -> i32
    "llvm.intr.assume"(%27) : (i1) -> ()
    %31 = llvm.getelementptr %24[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %32 = llvm.load %31 : !llvm.ptr -> i32
    llvm.br ^bb3(%6 : i64)
  ^bb3(%33: i64):  // 2 preds: ^bb2, ^bb4
    %34 = llvm.icmp "slt" %33, %19 : i64
    llvm.cond_br %34, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%15, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %35 = llvm.mlir.addressof @outOFL2L3_buff_0 : !llvm.ptr
    %36 = llvm.getelementptr %35[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<256 x i8>>>
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.and %37, %0  : i64
    %39 = llvm.icmp "eq" %38, %6 : i64
    "llvm.intr.assume"(%39) : (i1) -> ()
    %40 = llvm.mlir.addressof @act_4_5_buff_0 : !llvm.ptr
    %41 = llvm.getelementptr %40[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %0  : i64
    %44 = llvm.icmp "eq" %43, %6 : i64
    "llvm.intr.assume"(%44) : (i1) -> ()
    %45 = llvm.mlir.addressof @act_3_5_buff_0 : !llvm.ptr
    %46 = llvm.getelementptr %45[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.and %47, %0  : i64
    %49 = llvm.icmp "eq" %48, %6 : i64
    "llvm.intr.assume"(%49) : (i1) -> ()
    %50 = llvm.mlir.addressof @wts_buf_02_cons_buff_0 : !llvm.ptr
    %51 = llvm.getelementptr %50[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32768 x i8>
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.and %52, %0  : i64
    %54 = llvm.icmp "eq" %53, %6 : i64
    "llvm.intr.assume"(%54) : (i1) -> ()
    %55 = llvm.mlir.addressof @skip_buf_cons_buff_0 : !llvm.ptr
    %56 = llvm.getelementptr %55[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr to i64
    %58 = llvm.and %57, %0  : i64
    %59 = llvm.icmp "eq" %58, %6 : i64
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @conv2dk1_skip(%46, %41, %51, %36, %56, %4, %3, %2, %3, %28, %30, %32) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %18) : (i32, i32) -> ()
    %60 = llvm.mlir.addressof @outOFL2L3_buff_1 : !llvm.ptr
    %61 = llvm.getelementptr %60[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<256 x i8>>>
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.and %62, %0  : i64
    %64 = llvm.icmp "eq" %63, %6 : i64
    "llvm.intr.assume"(%64) : (i1) -> ()
    %65 = llvm.mlir.addressof @act_4_5_buff_1 : !llvm.ptr
    %66 = llvm.getelementptr %65[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %0  : i64
    %69 = llvm.icmp "eq" %68, %6 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.mlir.addressof @act_3_5_buff_1 : !llvm.ptr
    %71 = llvm.getelementptr %70[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %73 = llvm.and %72, %0  : i64
    %74 = llvm.icmp "eq" %73, %6 : i64
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    %75 = llvm.mlir.addressof @skip_buf_cons_buff_1 : !llvm.ptr
    %76 = llvm.getelementptr %75[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %77 = llvm.ptrtoint %76 : !llvm.ptr to i64
    %78 = llvm.and %77, %0  : i64
    %79 = llvm.icmp "eq" %78, %6 : i64
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @conv2dk1_skip(%71, %66, %51, %61, %76, %4, %3, %2, %3, %28, %30, %32) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %17) : (i32, i32) -> ()
    %80 = llvm.add %33, %20  : i64
    llvm.br ^bb3(%80 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%7, %17) : (i32, i32) -> ()
    %81 = llvm.add %21, %5  : i64
    llvm.br ^bb1(%81 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_5() {
    %0 = llvm.mlir.constant(31 : index) : i64
    %1 = llvm.mlir.constant(4294967292 : index) : i64
    %2 = llvm.mlir.constant(11 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant(64 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mlir.constant(48 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(53 : i32) : i32
    %12 = llvm.mlir.constant(52 : i32) : i32
    %13 = llvm.mlir.constant(51 : i32) : i32
    %14 = llvm.mlir.constant(49 : i32) : i32
    %15 = llvm.mlir.constant(28 : index) : i64
    %16 = llvm.mlir.constant(-2 : i32) : i32
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%19 : i64)
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb14
    %21 = llvm.icmp "slt" %20, %1 : i64
    llvm.cond_br %21, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %22 = llvm.mlir.addressof @act_4_5_buff_0 : !llvm.ptr
    %23 = llvm.getelementptr %22[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.and %24, %0  : i64
    %26 = llvm.icmp "eq" %25, %19 : i64
    "llvm.intr.assume"(%26) : (i1) -> ()
    %27 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_0 : !llvm.ptr
    %28 = llvm.getelementptr %27[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.and %29, %0  : i64
    %31 = llvm.icmp "eq" %30, %19 : i64
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    %32 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_1 : !llvm.ptr
    %33 = llvm.getelementptr %32[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %0  : i64
    %36 = llvm.icmp "eq" %35, %19 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.mlir.addressof @wts_buf_01_1_cons_buff_0 : !llvm.ptr
    %38 = llvm.getelementptr %37[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36864 x i8>
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.and %39, %0  : i64
    %41 = llvm.icmp "eq" %40, %19 : i64
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb3(%19 : i64)
  ^bb3(%42: i64):  // 2 preds: ^bb2, ^bb4
    %43 = llvm.icmp "slt" %42, %15 : i64
    llvm.cond_br %43, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %44 = llvm.mlir.addressof @act_4_5_buff_1 : !llvm.ptr
    %45 = llvm.getelementptr %44[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.and %46, %0  : i64
    %48 = llvm.icmp "eq" %47, %19 : i64
    "llvm.intr.assume"(%48) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %49 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_2 : !llvm.ptr
    %50 = llvm.getelementptr %49[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.and %51, %0  : i64
    %53 = llvm.icmp "eq" %52, %19 : i64
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %50, %38, %45, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    %54 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_3 : !llvm.ptr
    %55 = llvm.getelementptr %54[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %0  : i64
    %58 = llvm.icmp "eq" %57, %19 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %50, %55, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%48) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%50, %55, %28, %38, %45, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%55, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %59 = llvm.add %42, %18  : i64
    llvm.br ^bb3(%59 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %60 = llvm.mlir.addressof @act_4_5_buff_1 : !llvm.ptr
    %61 = llvm.getelementptr %60[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.and %62, %0  : i64
    %64 = llvm.icmp "eq" %63, %19 : i64
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %65 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_2 : !llvm.ptr
    %66 = llvm.getelementptr %65[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %0  : i64
    %69 = llvm.icmp "eq" %68, %19 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_3 : !llvm.ptr
    %71 = llvm.getelementptr %70[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %73 = llvm.and %72, %0  : i64
    %74 = llvm.icmp "eq" %73, %19 : i64
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb6(%19 : i64)
  ^bb6(%75: i64):  // 2 preds: ^bb5, ^bb7
    %76 = llvm.icmp "slt" %75, %15 : i64
    llvm.cond_br %76, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %28, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%71, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %77 = llvm.add %75, %18  : i64
    llvm.br ^bb6(%77 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb9(%19 : i64)
  ^bb9(%78: i64):  // 2 preds: ^bb8, ^bb10
    %79 = llvm.icmp "slt" %78, %15 : i64
    llvm.cond_br %79, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %28, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%71, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %80 = llvm.add %78, %18  : i64
    llvm.br ^bb9(%80 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb12(%19 : i64)
  ^bb12(%81: i64):  // 2 preds: ^bb11, ^bb13
    %82 = llvm.icmp "slt" %81, %15 : i64
    llvm.cond_br %82, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %28, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%71, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %83 = llvm.add %81, %18  : i64
    llvm.br ^bb12(%83 : i64)
  ^bb14:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    %84 = llvm.add %20, %18  : i64
    llvm.br ^bb1(%84 : i64)
  ^bb15:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %85 = llvm.mlir.addressof @act_4_5_buff_0 : !llvm.ptr
    %86 = llvm.getelementptr %85[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %87 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %88 = llvm.and %87, %0  : i64
    %89 = llvm.icmp "eq" %88, %19 : i64
    "llvm.intr.assume"(%89) : (i1) -> ()
    %90 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_0 : !llvm.ptr
    %91 = llvm.getelementptr %90[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.and %92, %0  : i64
    %94 = llvm.icmp "eq" %93, %19 : i64
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    %95 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_1 : !llvm.ptr
    %96 = llvm.getelementptr %95[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.and %97, %0  : i64
    %99 = llvm.icmp "eq" %98, %19 : i64
    "llvm.intr.assume"(%99) : (i1) -> ()
    %100 = llvm.mlir.addressof @wts_buf_01_1_cons_buff_0 : !llvm.ptr
    %101 = llvm.getelementptr %100[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36864 x i8>
    %102 = llvm.ptrtoint %101 : !llvm.ptr to i64
    %103 = llvm.and %102, %0  : i64
    %104 = llvm.icmp "eq" %103, %19 : i64
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %91, %96, %101, %86, %8, %7, %8, %6, %6, %5, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb16(%19 : i64)
  ^bb16(%105: i64):  // 2 preds: ^bb15, ^bb17
    %106 = llvm.icmp "slt" %105, %15 : i64
    llvm.cond_br %106, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %107 = llvm.mlir.addressof @act_4_5_buff_1 : !llvm.ptr
    %108 = llvm.getelementptr %107[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.and %109, %0  : i64
    %111 = llvm.icmp "eq" %110, %19 : i64
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    %112 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_2 : !llvm.ptr
    %113 = llvm.getelementptr %112[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %114 = llvm.ptrtoint %113 : !llvm.ptr to i64
    %115 = llvm.and %114, %0  : i64
    %116 = llvm.icmp "eq" %115, %19 : i64
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %113, %101, %108, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    %117 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_3 : !llvm.ptr
    %118 = llvm.getelementptr %117[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %119 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %120 = llvm.and %119, %0  : i64
    %121 = llvm.icmp "eq" %120, %19 : i64
    "llvm.intr.assume"(%121) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %113, %118, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%121) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%113, %118, %91, %101, %108, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%121) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%118, %91, %96, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %122 = llvm.add %105, %18  : i64
    llvm.br ^bb16(%122 : i64)
  ^bb18:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %123 = llvm.mlir.addressof @act_4_5_buff_1 : !llvm.ptr
    %124 = llvm.getelementptr %123[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
    %126 = llvm.and %125, %0  : i64
    %127 = llvm.icmp "eq" %126, %19 : i64
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    %128 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_2 : !llvm.ptr
    %129 = llvm.getelementptr %128[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %130 = llvm.ptrtoint %129 : !llvm.ptr to i64
    %131 = llvm.and %130, %0  : i64
    %132 = llvm.icmp "eq" %131, %19 : i64
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    %133 = llvm.mlir.addressof @act_2_3_4_1_cons_buff_3 : !llvm.ptr
    %134 = llvm.getelementptr %133[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %135 = llvm.ptrtoint %134 : !llvm.ptr to i64
    %136 = llvm.and %135, %0  : i64
    %137 = llvm.icmp "eq" %136, %19 : i64
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %134, %101, %124, %8, %7, %8, %6, %6, %3, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %91, %96, %101, %86, %8, %7, %8, %6, %6, %5, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb19(%19 : i64)
  ^bb19(%138: i64):  // 2 preds: ^bb18, ^bb20
    %139 = llvm.icmp "slt" %138, %15 : i64
    llvm.cond_br %139, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %91, %101, %124, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%134, %91, %96, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %140 = llvm.add %138, %18  : i64
    llvm.br ^bb19(%140 : i64)
  ^bb21:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %134, %101, %124, %8, %7, %8, %6, %6, %3, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %91, %96, %101, %86, %8, %7, %8, %6, %6, %5, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb22(%19 : i64)
  ^bb22(%141: i64):  // 2 preds: ^bb21, ^bb23
    %142 = llvm.icmp "slt" %141, %15 : i64
    llvm.cond_br %142, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %91, %101, %124, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%134, %91, %96, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %143 = llvm.add %141, %18  : i64
    llvm.br ^bb22(%143 : i64)
  ^bb24:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %134, %101, %124, %8, %7, %8, %6, %6, %3, %2, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_3() {
    %0 = llvm.mlir.constant(31 : index) : i64
    %1 = llvm.mlir.constant(4294967292 : index) : i64
    %2 = llvm.mlir.constant(11 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant(64 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mlir.constant(48 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(53 : i32) : i32
    %12 = llvm.mlir.constant(52 : i32) : i32
    %13 = llvm.mlir.constant(51 : i32) : i32
    %14 = llvm.mlir.constant(49 : i32) : i32
    %15 = llvm.mlir.constant(28 : index) : i64
    %16 = llvm.mlir.constant(-2 : i32) : i32
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%19 : i64)
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb14
    %21 = llvm.icmp "slt" %20, %1 : i64
    llvm.cond_br %21, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %22 = llvm.mlir.addressof @act_3_5_buff_0 : !llvm.ptr
    %23 = llvm.getelementptr %22[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.and %24, %0  : i64
    %26 = llvm.icmp "eq" %25, %19 : i64
    "llvm.intr.assume"(%26) : (i1) -> ()
    %27 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_0 : !llvm.ptr
    %28 = llvm.getelementptr %27[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.and %29, %0  : i64
    %31 = llvm.icmp "eq" %30, %19 : i64
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    %32 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_1 : !llvm.ptr
    %33 = llvm.getelementptr %32[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %0  : i64
    %36 = llvm.icmp "eq" %35, %19 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.mlir.addressof @wts_buf_01_0_cons_buff_0 : !llvm.ptr
    %38 = llvm.getelementptr %37[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36864 x i8>
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.and %39, %0  : i64
    %41 = llvm.icmp "eq" %40, %19 : i64
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb3(%19 : i64)
  ^bb3(%42: i64):  // 2 preds: ^bb2, ^bb4
    %43 = llvm.icmp "slt" %42, %15 : i64
    llvm.cond_br %43, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %44 = llvm.mlir.addressof @act_3_5_buff_1 : !llvm.ptr
    %45 = llvm.getelementptr %44[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.and %46, %0  : i64
    %48 = llvm.icmp "eq" %47, %19 : i64
    "llvm.intr.assume"(%48) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %49 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_2 : !llvm.ptr
    %50 = llvm.getelementptr %49[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.and %51, %0  : i64
    %53 = llvm.icmp "eq" %52, %19 : i64
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %50, %38, %45, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    %54 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_3 : !llvm.ptr
    %55 = llvm.getelementptr %54[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %0  : i64
    %58 = llvm.icmp "eq" %57, %19 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %50, %55, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%48) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%50, %55, %28, %38, %45, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%55, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %59 = llvm.add %42, %18  : i64
    llvm.br ^bb3(%59 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %60 = llvm.mlir.addressof @act_3_5_buff_1 : !llvm.ptr
    %61 = llvm.getelementptr %60[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.and %62, %0  : i64
    %64 = llvm.icmp "eq" %63, %19 : i64
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %65 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_2 : !llvm.ptr
    %66 = llvm.getelementptr %65[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %0  : i64
    %69 = llvm.icmp "eq" %68, %19 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_3 : !llvm.ptr
    %71 = llvm.getelementptr %70[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %73 = llvm.and %72, %0  : i64
    %74 = llvm.icmp "eq" %73, %19 : i64
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb6(%19 : i64)
  ^bb6(%75: i64):  // 2 preds: ^bb5, ^bb7
    %76 = llvm.icmp "slt" %75, %15 : i64
    llvm.cond_br %76, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %28, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%71, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %77 = llvm.add %75, %18  : i64
    llvm.br ^bb6(%77 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb9(%19 : i64)
  ^bb9(%78: i64):  // 2 preds: ^bb8, ^bb10
    %79 = llvm.icmp "slt" %78, %15 : i64
    llvm.cond_br %79, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %28, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%71, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %80 = llvm.add %78, %18  : i64
    llvm.br ^bb9(%80 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %28, %33, %38, %23, %8, %7, %8, %6, %6, %5, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb12(%19 : i64)
  ^bb12(%81: i64):  // 2 preds: ^bb11, ^bb13
    %82 = llvm.icmp "slt" %81, %15 : i64
    llvm.cond_br %82, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %28, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%71, %28, %33, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %83 = llvm.add %81, %18  : i64
    llvm.br ^bb12(%83 : i64)
  ^bb14:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%31) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%28, %33, %66, %38, %61, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%33, %66, %71, %38, %23, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    "llvm.intr.assume"(%41) : (i1) -> ()
    llvm.call @conv2dk3(%66, %71, %71, %38, %61, %8, %7, %8, %6, %6, %3, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    %84 = llvm.add %20, %18  : i64
    llvm.br ^bb1(%84 : i64)
  ^bb15:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %85 = llvm.mlir.addressof @act_3_5_buff_0 : !llvm.ptr
    %86 = llvm.getelementptr %85[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %87 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %88 = llvm.and %87, %0  : i64
    %89 = llvm.icmp "eq" %88, %19 : i64
    "llvm.intr.assume"(%89) : (i1) -> ()
    %90 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_0 : !llvm.ptr
    %91 = llvm.getelementptr %90[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.and %92, %0  : i64
    %94 = llvm.icmp "eq" %93, %19 : i64
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    %95 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_1 : !llvm.ptr
    %96 = llvm.getelementptr %95[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.and %97, %0  : i64
    %99 = llvm.icmp "eq" %98, %19 : i64
    "llvm.intr.assume"(%99) : (i1) -> ()
    %100 = llvm.mlir.addressof @wts_buf_01_0_cons_buff_0 : !llvm.ptr
    %101 = llvm.getelementptr %100[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36864 x i8>
    %102 = llvm.ptrtoint %101 : !llvm.ptr to i64
    %103 = llvm.and %102, %0  : i64
    %104 = llvm.icmp "eq" %103, %19 : i64
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %91, %96, %101, %86, %8, %7, %8, %6, %6, %5, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb16(%19 : i64)
  ^bb16(%105: i64):  // 2 preds: ^bb15, ^bb17
    %106 = llvm.icmp "slt" %105, %15 : i64
    llvm.cond_br %106, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %107 = llvm.mlir.addressof @act_3_5_buff_1 : !llvm.ptr
    %108 = llvm.getelementptr %107[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.and %109, %0  : i64
    %111 = llvm.icmp "eq" %110, %19 : i64
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    %112 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_2 : !llvm.ptr
    %113 = llvm.getelementptr %112[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %114 = llvm.ptrtoint %113 : !llvm.ptr to i64
    %115 = llvm.and %114, %0  : i64
    %116 = llvm.icmp "eq" %115, %19 : i64
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %113, %101, %108, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    %117 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_3 : !llvm.ptr
    %118 = llvm.getelementptr %117[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %119 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %120 = llvm.and %119, %0  : i64
    %121 = llvm.icmp "eq" %120, %19 : i64
    "llvm.intr.assume"(%121) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %113, %118, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%121) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%113, %118, %91, %101, %108, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%121) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%118, %91, %96, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %122 = llvm.add %105, %18  : i64
    llvm.br ^bb16(%122 : i64)
  ^bb18:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    %123 = llvm.mlir.addressof @act_3_5_buff_1 : !llvm.ptr
    %124 = llvm.getelementptr %123[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<32 x i8>>>
    %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
    %126 = llvm.and %125, %0  : i64
    %127 = llvm.icmp "eq" %126, %19 : i64
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    %128 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_2 : !llvm.ptr
    %129 = llvm.getelementptr %128[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %130 = llvm.ptrtoint %129 : !llvm.ptr to i64
    %131 = llvm.and %130, %0  : i64
    %132 = llvm.icmp "eq" %131, %19 : i64
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    %133 = llvm.mlir.addressof @act_2_3_4_0_cons_buff_3 : !llvm.ptr
    %134 = llvm.getelementptr %133[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %135 = llvm.ptrtoint %134 : !llvm.ptr to i64
    %136 = llvm.and %135, %0  : i64
    %137 = llvm.icmp "eq" %136, %19 : i64
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %134, %101, %124, %8, %7, %8, %6, %6, %3, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %91, %96, %101, %86, %8, %7, %8, %6, %6, %5, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb19(%19 : i64)
  ^bb19(%138: i64):  // 2 preds: ^bb18, ^bb20
    %139 = llvm.icmp "slt" %138, %15 : i64
    llvm.cond_br %139, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %91, %101, %124, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%134, %91, %96, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %140 = llvm.add %138, %18  : i64
    llvm.br ^bb19(%140 : i64)
  ^bb21:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %134, %101, %124, %8, %7, %8, %6, %6, %3, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %91, %96, %101, %86, %8, %7, %8, %6, %6, %5, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.br ^bb22(%19 : i64)
  ^bb22(%141: i64):  // 2 preds: ^bb21, ^bb23
    %142 = llvm.icmp "slt" %141, %15 : i64
    llvm.cond_br %142, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %91, %101, %124, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%134, %91, %96, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    %143 = llvm.add %141, %18  : i64
    llvm.br ^bb22(%143 : i64)
  ^bb24:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%94) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%91, %96, %129, %101, %124, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%96, %129, %134, %101, %86, %8, %7, %8, %6, %6, %4, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%127) : (i1) -> ()
    "llvm.intr.assume"(%132) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%137) : (i1) -> ()
    "llvm.intr.assume"(%104) : (i1) -> ()
    llvm.call @conv2dk3(%129, %134, %134, %101, %124, %8, %7, %8, %6, %6, %3, %2, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %4) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %3) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %4) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_2() {
    %0 = llvm.mlir.constant(31 : index) : i64
    %1 = llvm.mlir.constant(4294967295 : index) : i64
    %2 = llvm.mlir.constant(64 : i32) : i32
    %3 = llvm.mlir.constant(32 : i32) : i32
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(50 : i32) : i32
    %6 = llvm.mlir.constant(53 : i32) : i32
    %7 = llvm.mlir.constant(48 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(49 : i32) : i32
    %10 = llvm.mlir.constant(51 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(2 : index) : i64
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(32 : index) : i64
    %15 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb5
    %17 = llvm.icmp "slt" %16, %1 : i64
    llvm.cond_br %17, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    %18 = llvm.mlir.addressof @rtp2 : !llvm.ptr
    %19 = llvm.getelementptr %18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.and %20, %0  : i64
    %22 = llvm.icmp "eq" %21, %15 : i64
    "llvm.intr.assume"(%22) : (i1) -> ()
    %23 = llvm.load %19 : !llvm.ptr -> i32
    llvm.br ^bb3(%15 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb4
    %25 = llvm.icmp "slt" %24, %14 : i64
    llvm.cond_br %25, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    %26 = llvm.mlir.addressof @act_2_3_4_buff_0 : !llvm.ptr
    %27 = llvm.getelementptr %26[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.and %28, %0  : i64
    %30 = llvm.icmp "eq" %29, %15 : i64
    "llvm.intr.assume"(%30) : (i1) -> ()
    %31 = llvm.mlir.addressof @wts_buf_00_cons_buff_0 : !llvm.ptr
    %32 = llvm.getelementptr %31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4096 x i8>
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.and %33, %0  : i64
    %35 = llvm.icmp "eq" %34, %15 : i64
    "llvm.intr.assume"(%35) : (i1) -> ()
    %36 = llvm.mlir.addressof @inOF_act_L3L2_0_cons_buff_0 : !llvm.ptr
    %37 = llvm.getelementptr %36[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.and %38, %0  : i64
    %40 = llvm.icmp "eq" %39, %15 : i64
    "llvm.intr.assume"(%40) : (i1) -> ()
    llvm.call @conv2dk1(%37, %32, %27, %3, %2, %2, %23) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    %41 = llvm.mlir.addressof @act_2_3_4_buff_1 : !llvm.ptr
    %42 = llvm.getelementptr %41[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.and %43, %0  : i64
    %45 = llvm.icmp "eq" %44, %15 : i64
    "llvm.intr.assume"(%45) : (i1) -> ()
    "llvm.intr.assume"(%35) : (i1) -> ()
    %46 = llvm.mlir.addressof @inOF_act_L3L2_0_cons_buff_1 : !llvm.ptr
    %47 = llvm.getelementptr %46[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<1 x array<64 x i8>>>
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.and %48, %0  : i64
    %50 = llvm.icmp "eq" %49, %15 : i64
    "llvm.intr.assume"(%50) : (i1) -> ()
    llvm.call @conv2dk1(%47, %32, %42, %3, %2, %2, %23) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %11) : (i32, i32) -> ()
    %51 = llvm.add %24, %12  : i64
    llvm.br ^bb3(%51 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%5, %11) : (i32, i32) -> ()
    %52 = llvm.add %16, %4  : i64
    llvm.br ^bb1(%52 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}

