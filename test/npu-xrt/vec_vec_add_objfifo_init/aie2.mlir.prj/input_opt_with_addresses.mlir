module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @in1_buff_0(dense<"0x0100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF00000000010000"> : tensor<256xi32>) {addr_space = 0 : i32} : !llvm.array<256 x i32>
  llvm.mlir.global external @in1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<256 x i32>
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
  llvm.mlir.global external @in1_cons() {addr_space = 0 : i32} : !llvm.array<256 x i32>
  llvm.mlir.global external @in1() {addr_space = 0 : i32} : !llvm.array<256 x i32>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @out_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @in2_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @out_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @in2_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @in1_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(48 : i32) : i32
    %6 = llvm.mlir.constant(53 : i32) : i32
    %7 = llvm.mlir.constant(50 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(49 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(2 : index) : i64
    %13 = llvm.mlir.constant(16 : index) : i64
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb11
    %19 = llvm.icmp "slt" %18, %16 : i64
    llvm.cond_br %19, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%10, %14) : (i32, i32) -> ()
    llvm.br ^bb3(%15 : i64)
  ^bb3(%20: i64):  // 2 preds: ^bb2, ^bb10
    %21 = llvm.icmp "slt" %20, %13 : i64
    llvm.cond_br %21, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %14) : (i32, i32) -> ()
    llvm.br ^bb5(%15 : i64)
  ^bb5(%22: i64):  // 2 preds: ^bb4, ^bb6
    %23 = llvm.icmp "slt" %22, %13 : i64
    llvm.cond_br %23, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %24 = llvm.mul %20, %13 : i64
    %25 = llvm.add %24, %22 : i64
    %26 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x i32>
    %27 = llvm.getelementptr inbounds|nuw %26[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %28 = llvm.load %27 : !llvm.ptr -> i32
    %29 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %30 = llvm.getelementptr inbounds|nuw %29[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %31 = llvm.load %30 : !llvm.ptr -> i32
    %32 = llvm.add %28, %31 : i32
    %33 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %34 = llvm.getelementptr inbounds|nuw %33[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %32, %34 : i32, !llvm.ptr
    %35 = llvm.add %22, %17 : i64
    llvm.br ^bb5(%35 : i64)
  ^bb7:  // pred: ^bb5
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %11) : (i32, i32) -> ()
    %36 = llvm.add %20, %17 : i64
    llvm.call @llvm.aie2.acquire(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %14) : (i32, i32) -> ()
    llvm.br ^bb8(%15 : i64)
  ^bb8(%37: i64):  // 2 preds: ^bb7, ^bb9
    %38 = llvm.icmp "slt" %37, %13 : i64
    llvm.cond_br %38, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %39 = llvm.mul %36, %13 : i64
    %40 = llvm.add %39, %37 : i64
    %41 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x i32>
    %42 = llvm.getelementptr inbounds|nuw %41[%40] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %43 = llvm.load %42 : !llvm.ptr -> i32
    %44 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %45 = llvm.getelementptr inbounds|nuw %44[%37] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %46 = llvm.load %45 : !llvm.ptr -> i32
    %47 = llvm.add %43, %46 : i32
    %48 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %49 = llvm.getelementptr inbounds|nuw %48[%37] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %47, %49 : i32, !llvm.ptr
    %50 = llvm.add %37, %17 : i64
    llvm.br ^bb8(%50 : i64)
  ^bb10:  // pred: ^bb8
    llvm.call @llvm.aie2.release(%7, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %11) : (i32, i32) -> ()
    %51 = llvm.add %20, %12 : i64
    llvm.br ^bb3(%51 : i64)
  ^bb11:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%5, %11) : (i32, i32) -> ()
    %52 = llvm.add %18, %17 : i64
    llvm.br ^bb1(%52 : i64)
  ^bb12:  // pred: ^bb1
    llvm.return
  }
}

