module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @input_fifo_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.mlir.global external @input_fifo_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.mlir.global external @input_fifo_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.mlir.global external @output_fifo_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.mlir.global external @output_fifo_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @output_fifo_cons() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.mlir.global external @output_fifo() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.mlir.global external @input_fifo_cons() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.mlir.global external @input_fifo() {addr_space = 0 : i32} : !llvm.array<64 x i32>
  llvm.func @sum_64_i32(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @sequence(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @output_fifo_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @output_fifo_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @input_fifo_cons_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @input_fifo_cons_buff_1 : !llvm.ptr
    %4 = llvm.mlir.constant(31 : index) : i64
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.constant(3735928559 : index) : i64
    %7 = llvm.mlir.addressof @input_fifo_cons_buff_0 : !llvm.ptr
    %8 = llvm.mlir.constant(64 : index) : i64
    %9 = llvm.mlir.constant(3 : index) : i64
    %10 = llvm.mlir.constant(2 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(4294967295 : index) : i64
    %13 = llvm.mlir.constant(51 : i32) : i32
    %14 = llvm.mlir.constant(48 : i32) : i32
    %15 = llvm.mlir.constant(49 : i32) : i32
    %16 = llvm.mlir.constant(4294967294 : index) : i64
    %17 = llvm.mlir.constant(50 : i32) : i32
    %18 = llvm.mlir.constant(2 : i32) : i32
    %19 = llvm.mlir.constant(1 : i32) : i32
    %20 = llvm.mlir.constant(-2 : i32) : i32
    %21 = llvm.mlir.constant(-1 : i32) : i32
    %22 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%22, %22 : i64, i64)
  ^bb1(%23: i64, %24: i64):  // 2 preds: ^bb0, ^bb17
    %25 = llvm.icmp "slt" %23, %12 : i64
    llvm.cond_br %25, ^bb2, ^bb18
  ^bb2:  // pred: ^bb1
    %26 = llvm.srem %23, %10  : i64
    %27 = llvm.icmp "eq" %26, %22 : i64
    %28 = llvm.srem %24, %9  : i64
    %29 = llvm.trunc %28 : i64 to i32
    llvm.switch %29 : i32, ^bb3 [
      0: ^bb3,
      1: ^bb4,
      2: ^bb5
    ]
  ^bb3:  // 2 preds: ^bb2, ^bb2
    %30 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %31 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %32 = llvm.insertvalue %31, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %22, %33[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %8, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %11, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %38 = llvm.and %37, %4  : i64
    %39 = llvm.icmp "eq" %38, %22 : i64
    "llvm.intr.assume"(%39) : (i1) -> ()
    llvm.br ^bb6(%36 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb4:  // pred: ^bb2
    %40 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %41 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %42 = llvm.insertvalue %41, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.insertvalue %40, %42[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %22, %43[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %8, %44[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %11, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %48 = llvm.and %47, %4  : i64
    %49 = llvm.icmp "eq" %48, %22 : i64
    "llvm.intr.assume"(%49) : (i1) -> ()
    llvm.br ^bb6(%46 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb5:  // pred: ^bb2
    %50 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %51 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %52 = llvm.insertvalue %51, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %22, %53[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %8, %54[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %11, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %58 = llvm.and %57, %4  : i64
    %59 = llvm.icmp "eq" %58, %22 : i64
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.br ^bb6(%56 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb6(%60: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 3 preds: ^bb3, ^bb4, ^bb5
    llvm.switch %29 : i32, ^bb9 [
      0: ^bb7,
      1: ^bb8,
      2: ^bb9
    ]
  ^bb7:  // pred: ^bb6
    %61 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %62 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %63 = llvm.insertvalue %62, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %64 = llvm.insertvalue %61, %63[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %22, %64[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %8, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %11, %66[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %69 = llvm.and %68, %4  : i64
    %70 = llvm.icmp "eq" %69, %22 : i64
    "llvm.intr.assume"(%70) : (i1) -> ()
    llvm.br ^bb10(%67 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb8:  // pred: ^bb6
    %71 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %72 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %73 = llvm.insertvalue %72, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.insertvalue %71, %73[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.insertvalue %22, %74[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.insertvalue %8, %75[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.insertvalue %11, %76[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %79 = llvm.and %78, %4  : i64
    %80 = llvm.icmp "eq" %79, %22 : i64
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.br ^bb10(%77 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb9:  // 2 preds: ^bb6, ^bb6
    %81 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %82 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %83 = llvm.insertvalue %82, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.insertvalue %81, %83[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.insertvalue %22, %84[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %8, %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.insertvalue %11, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %89 = llvm.and %88, %4  : i64
    %90 = llvm.icmp "eq" %89, %22 : i64
    "llvm.intr.assume"(%90) : (i1) -> ()
    llvm.br ^bb10(%87 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb10(%91: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 3 preds: ^bb7, ^bb8, ^bb9
    llvm.cond_br %27, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %92 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %93 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %94 = llvm.insertvalue %93, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.insertvalue %22, %95[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %97 = llvm.insertvalue %8, %96[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.insertvalue %11, %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.ptrtoint %92 : !llvm.ptr to i64
    %100 = llvm.and %99, %4  : i64
    %101 = llvm.icmp "eq" %100, %22 : i64
    "llvm.intr.assume"(%101) : (i1) -> ()
    llvm.br ^bb13(%98 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb12:  // pred: ^bb10
    %102 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i32>
    %103 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %104 = llvm.insertvalue %103, %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.insertvalue %22, %105[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = llvm.insertvalue %8, %106[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %108 = llvm.insertvalue %11, %107[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %109 = llvm.ptrtoint %102 : !llvm.ptr to i64
    %110 = llvm.and %109, %4  : i64
    %111 = llvm.icmp "eq" %110, %22 : i64
    "llvm.intr.assume"(%111) : (i1) -> ()
    llvm.br ^bb13(%108 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb13(%112: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb11, ^bb12
    llvm.call @llvm.aie2.acquire(%17, %21) : (i32, i32) -> ()
    %113 = llvm.icmp "eq" %23, %22 : i64
    %114 = llvm.icmp "eq" %23, %16 : i64
    llvm.cond_br %113, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    llvm.call @llvm.aie2.acquire(%15, %21) : (i32, i32) -> ()
    %115 = llvm.extractvalue %60[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %116 = llvm.extractvalue %112[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @sum_64_i32(%115, %115, %116) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb17(%22 : i64)
  ^bb15:  // pred: ^bb13
    llvm.cond_br %114, ^bb16(%20, %91, %19, %11 : i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i32, i64), ^bb16(%21, %60, %18, %10 : i32, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i32, i64)
  ^bb16(%117: i32, %118: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, %119: i32, %120: i64):  // 2 preds: ^bb15, ^bb15
    llvm.call @llvm.aie2.acquire(%15, %117) : (i32, i32) -> ()
    %121 = llvm.extractvalue %60[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %118[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.extractvalue %112[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @sum_64_i32(%121, %122, %123) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%14, %119) : (i32, i32) -> ()
    llvm.br ^bb17(%120 : i64)
  ^bb17(%124: i64):  // 2 preds: ^bb14, ^bb16
    llvm.call @llvm.aie2.release(%13, %19) : (i32, i32) -> ()
    %125 = llvm.add %23, %11 : i64
    llvm.br ^bb1(%125, %124 : i64, i64)
  ^bb18:  // pred: ^bb1
    llvm.return
  }
}

