module @stencil_2d_7point attributes {llvm.data_layout = "", llvm.target_triple = "aie"} {
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
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
  llvm.mlir.global external @of_1_buff_0() {addr_space = 0 : i32} : !llvm.array<256 x f32>
  llvm.mlir.global external @of_1_buff_1() {addr_space = 0 : i32} : !llvm.array<256 x f32>
  llvm.mlir.global external @of_1_buff_2() {addr_space = 0 : i32} : !llvm.array<256 x f32>
  llvm.mlir.global external @of_1_buff_3() {addr_space = 0 : i32} : !llvm.array<256 x f32>
  llvm.mlir.global external @of_2_buff_0() {addr_space = 0 : i32} : !llvm.array<256 x f32>
  llvm.mlir.global external @of_2_buff_1() {addr_space = 0 : i32} : !llvm.array<256 x f32>
  llvm.func @stencil_2d_7point_fp32(!llvm.ptr<f32>, !llvm.ptr<f32>, !llvm.ptr<f32>, !llvm.ptr<f32>) attributes {sym_visibility = "private"}
  llvm.func @core_7_1() {
    %0 = llvm.mlir.constant(31 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(62 : i32) : i32
    %3 = llvm.mlir.constant(52 : i32) : i32
    %4 = llvm.mlir.constant(48 : i32) : i32
    %5 = llvm.mlir.constant(50 : i32) : i32
    %6 = llvm.mlir.constant(49 : i32) : i32
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    llvm.call @llvm.aie.lock.acquire.reg(%2, %8) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.acquire.reg(%4, %7) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.acquire.reg(%6, %7) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.acquire.reg(%5, %7) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.acquire.reg(%3, %8) : (i32, i32) -> ()
    %9 = llvm.mlir.addressof @of_1_buff_0 : !llvm.ptr<array<256 x f32>>
    %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr<array<256 x f32>>) -> !llvm.ptr<f32>
    %11 = llvm.ptrtoint %10 : !llvm.ptr<f32> to i64
    %12 = llvm.and %11, %0  : i64
    %13 = llvm.icmp "eq" %12, %1 : i64
    "llvm.intr.assume"(%13) : (i1) -> ()
    %14 = llvm.mlir.addressof @of_1_buff_1 : !llvm.ptr<array<256 x f32>>
    %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr<array<256 x f32>>) -> !llvm.ptr<f32>
    %16 = llvm.ptrtoint %15 : !llvm.ptr<f32> to i64
    %17 = llvm.and %16, %0  : i64
    %18 = llvm.icmp "eq" %17, %1 : i64
    "llvm.intr.assume"(%18) : (i1) -> ()
    %19 = llvm.mlir.addressof @of_1_buff_2 : !llvm.ptr<array<256 x f32>>
    %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr<array<256 x f32>>) -> !llvm.ptr<f32>
    %21 = llvm.ptrtoint %20 : !llvm.ptr<f32> to i64
    %22 = llvm.and %21, %0  : i64
    %23 = llvm.icmp "eq" %22, %1 : i64
    "llvm.intr.assume"(%23) : (i1) -> ()
    %24 = llvm.mlir.addressof @of_2_buff_0 : !llvm.ptr<array<256 x f32>>
    %25 = llvm.getelementptr %24[0, 0] : (!llvm.ptr<array<256 x f32>>) -> !llvm.ptr<f32>
    %26 = llvm.ptrtoint %25 : !llvm.ptr<f32> to i64
    %27 = llvm.and %26, %0  : i64
    %28 = llvm.icmp "eq" %27, %1 : i64
    "llvm.intr.assume"(%28) : (i1) -> ()
    llvm.call @stencil_2d_7point_fp32(%10, %15, %20, %25) : (!llvm.ptr<f32>, !llvm.ptr<f32>, !llvm.ptr<f32>, !llvm.ptr<f32>) -> ()
    llvm.call @llvm.aie.lock.release.reg(%4, %8) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.release.reg(%3, %7) : (i32, i32) -> ()
    llvm.call @llvm.aie.lock.release.reg(%2, %8) : (i32, i32) -> ()
    llvm.return
  }
}

