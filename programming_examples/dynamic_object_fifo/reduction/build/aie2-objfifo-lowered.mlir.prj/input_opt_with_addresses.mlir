module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @input_fifo_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @input_fifo_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @input_fifo_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @input_fifo_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @output_fifo_buff_1() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @output_fifo_buff_0() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @output_fifo_cons() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @output_fifo() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @input_fifo_cons() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external @input_fifo() {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.func @sum_10_i32(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @output_fifo_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @output_fifo_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @input_fifo_cons_buff_3 : !llvm.ptr
    %3 = llvm.mlir.addressof @input_fifo_cons_buff_1 : !llvm.ptr
    %4 = llvm.mlir.addressof @input_fifo_cons_buff_2 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.mlir.constant(3735928559 : index) : i64
    %8 = llvm.mlir.addressof @input_fifo_cons_buff_0 : !llvm.ptr
    %9 = llvm.mlir.constant(2 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(10 : index) : i64
    %12 = llvm.mlir.constant(48 : i32) : i32
    %13 = llvm.mlir.constant(51 : i32) : i32
    %14 = llvm.mlir.constant(49 : i32) : i32
    %15 = llvm.mlir.constant(50 : i32) : i32
    %16 = llvm.mlir.constant(2 : i32) : i32
    %17 = llvm.mlir.constant(1 : i32) : i32
    %18 = llvm.mlir.constant(-2 : i32) : i32
    %19 = llvm.mlir.constant(-1 : i32) : i32
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20, %20, %20 : i64, i64, i64)
  ^bb1(%21: i64, %22: i64, %23: i64):  // 2 preds: ^bb0, ^bb11
    %24 = llvm.icmp "slt" %21, %11 : i64
    llvm.cond_br %24, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %25 = llvm.srem %22, %9  : i64
    %26 = llvm.srem %23, %9  : i64
    %27 = llvm.trunc %25 : i64 to i32
    llvm.switch %27 : i32, ^bb3 [
      0: ^bb3,
      1: ^bb4
    ]
  ^bb3:  // 2 preds: ^bb2, ^bb2
    %28 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
    %29 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %30 = llvm.insertvalue %29, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %20, %31[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %11, %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %10, %33[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %36 = llvm.and %35, %5  : i64
    %37 = llvm.icmp "eq" %36, %20 : i64
    "llvm.intr.assume"(%37) : (i1) -> ()
    llvm.br ^bb5(%34 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb4:  // pred: ^bb2
    %38 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
    %39 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %40 = llvm.insertvalue %39, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.insertvalue %38, %40[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.insertvalue %20, %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.insertvalue %11, %42[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %10, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %46 = llvm.and %45, %5  : i64
    %47 = llvm.icmp "eq" %46, %20 : i64
    "llvm.intr.assume"(%47) : (i1) -> ()
    llvm.br ^bb5(%44 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb5(%48: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb3, ^bb4
    llvm.switch %27 : i32, ^bb6 [
      0: ^bb6,
      1: ^bb7
    ]
  ^bb6:  // 2 preds: ^bb5, ^bb5
    %49 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
    %50 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %51 = llvm.insertvalue %50, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.insertvalue %20, %52[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %11, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %10, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %57 = llvm.and %56, %5  : i64
    %58 = llvm.icmp "eq" %57, %20 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.br ^bb8(%55 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb7:  // pred: ^bb5
    %59 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
    %60 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %61 = llvm.insertvalue %60, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = llvm.insertvalue %59, %61[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %63 = llvm.insertvalue %20, %62[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %64 = llvm.insertvalue %11, %63[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %10, %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %67 = llvm.and %66, %5  : i64
    %68 = llvm.icmp "eq" %67, %20 : i64
    "llvm.intr.assume"(%68) : (i1) -> ()
    llvm.br ^bb8(%65 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb8(%69: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb6, ^bb7
    %70 = llvm.trunc %26 : i64 to i32
    llvm.switch %70 : i32, ^bb10 [
      0: ^bb9,
      1: ^bb10
    ]
  ^bb9:  // pred: ^bb8
    %71 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
    %72 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %73 = llvm.insertvalue %72, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.insertvalue %71, %73[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.insertvalue %20, %74[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.insertvalue %11, %75[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.insertvalue %10, %76[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %79 = llvm.and %78, %5  : i64
    %80 = llvm.icmp "eq" %79, %20 : i64
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.br ^bb11(%77 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb10:  // 2 preds: ^bb8, ^bb8
    %81 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
    %82 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %83 = llvm.insertvalue %82, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.insertvalue %81, %83[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.insertvalue %20, %84[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %11, %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.insertvalue %10, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %89 = llvm.and %88, %5  : i64
    %90 = llvm.icmp "eq" %89, %20 : i64
    "llvm.intr.assume"(%90) : (i1) -> ()
    llvm.br ^bb11(%87 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb11(%91: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb9, ^bb10
    llvm.call @llvm.aie2.acquire(%15, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %18) : (i32, i32) -> ()
    %92 = llvm.extractvalue %48[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %93 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %94 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @sum_10_i32(%92, %93, %94) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%13, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %16) : (i32, i32) -> ()
    %95 = llvm.add %22, %10 : i64
    %96 = llvm.add %23, %10 : i64
    %97 = llvm.add %21, %10 : i64
    llvm.br ^bb1(%97, %95, %96 : i64, i64, i64)
  ^bb12:  // pred: ^bb1
    llvm.return
  }
}

