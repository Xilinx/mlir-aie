module attributes {llvm.target_triple = "aie"} {
  llvm.mlir.global external @in_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @out_buff_1() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @out_buff_0() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.event0() attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.event1() attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.wms(i32, i128) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.fms(i32, f32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.ss(i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.wss(i32) -> i128 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.fss(i32) -> f32 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.mcd(i384) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.scd() -> i384 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.lock.acquire.reg(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.lock.release.reg(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @out_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @out() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in_cons() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @in() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.func @sequence(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.return
  }
  llvm.func @core_6_2() {
    %0 = llvm.mlir.constant(31 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %3 = llvm.mlir.constant(19 : i32) : i32
    %4 = llvm.mlir.constant(17 : i32) : i32
    %5 = llvm.mlir.constant(18 : i32) : i32
    %6 = llvm.mlir.constant(16 : i32) : i32
    %7 = llvm.mlir.constant(3 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(2 : index) : i64
    %11 = llvm.mlir.constant(4 : index) : i64
    %12 = llvm.mlir.constant(16 : index) : i64
    %13 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%13 : i64)
  ^bb1(%14: i64):  // 2 preds: ^bb0, ^bb10
    %15 = llvm.icmp "slt" %14, %2 : i64
    llvm.cond_br %15, ^bb2(%13 : i64), ^bb11
  ^bb2(%16: i64):  // 2 preds: ^bb1, ^bb9
    %17 = llvm.icmp "slt" %16, %11 : i64
    llvm.cond_br %17, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie.lock.acquire.reg(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.acquire.reg(%5, %8) : (i32, i32) -> ()
    llvm.br ^bb4(%13 : i64)
  ^bb4(%18: i64):  // 2 preds: ^bb3, ^bb5
    %19 = llvm.icmp "slt" %18, %12 : i64
    llvm.cond_br %19, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %20 = llvm.mlir.addressof @in_cons_buff_0 : !llvm.ptr
    %21 = llvm.getelementptr %20[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.and %22, %0  : i64
    %24 = llvm.icmp "eq" %23, %13 : i64
    "llvm.intr.assume"(%24) : (i1) -> ()
    %25 = llvm.getelementptr %21[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %26 = llvm.load %25 : !llvm.ptr -> i32
    %27 = llvm.mul %26, %7  : i32
    %28 = llvm.mlir.addressof @out_buff_0 : !llvm.ptr
    %29 = llvm.getelementptr %28[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.and %30, %0  : i64
    %32 = llvm.icmp "eq" %31, %13 : i64
    "llvm.intr.assume"(%32) : (i1) -> ()
    %33 = llvm.getelementptr %29[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %27, %33 : i32, !llvm.ptr
    %34 = llvm.add %18, %1  : i64
    llvm.br ^bb4(%34 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie.lock.release.reg(%6, %8) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.release.reg(%5, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.acquire.reg(%4, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.acquire.reg(%3, %8) : (i32, i32) -> ()
    llvm.br ^bb7(%13 : i64)
  ^bb7(%35: i64):  // 2 preds: ^bb6, ^bb8
    %36 = llvm.icmp "slt" %35, %12 : i64
    llvm.cond_br %36, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %37 = llvm.mlir.addressof @in_cons_buff_1 : !llvm.ptr
    %38 = llvm.getelementptr %37[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.and %39, %0  : i64
    %41 = llvm.icmp "eq" %40, %13 : i64
    "llvm.intr.assume"(%41) : (i1) -> ()
    %42 = llvm.getelementptr %38[%35] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %43 = llvm.load %42 : !llvm.ptr -> i32
    %44 = llvm.mul %43, %7  : i32
    %45 = llvm.mlir.addressof @out_buff_1 : !llvm.ptr
    %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.and %47, %0  : i64
    %49 = llvm.icmp "eq" %48, %13 : i64
    "llvm.intr.assume"(%49) : (i1) -> ()
    %50 = llvm.getelementptr %46[%35] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %44, %50 : i32, !llvm.ptr
    %51 = llvm.add %35, %1  : i64
    llvm.br ^bb7(%51 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie.lock.release.reg(%4, %8) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.release.reg(%3, %9) : (i32, i32) -> ()
    %52 = llvm.add %16, %10  : i64
    llvm.br ^bb2(%52 : i64)
  ^bb10:  // pred: ^bb2
    %53 = llvm.add %14, %1  : i64
    llvm.br ^bb1(%53 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
}

