; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@A_L2L1_0_1_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_0_1_cons_buff_0 = external global [16 x [32 x i16]]
@A_L2L1_0_0_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_0_0_cons_buff_0 = external global [16 x [32 x i16]]
@A_L2L1_1_1_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_1_1_cons_buff_0 = external global [16 x [32 x i16]]
@A_L2L1_1_0_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_1_0_cons_buff_0 = external global [16 x [32 x i16]]
@A_L2L1_2_1_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_2_1_cons_buff_0 = external global [16 x [32 x i16]]
@A_L2L1_2_0_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_2_0_cons_buff_0 = external global [16 x [32 x i16]]
@A_L2L1_3_1_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_3_1_cons_buff_0 = external global [16 x [32 x i16]]
@A_L2L1_3_0_cons_buff_1 = external global [16 x [32 x i16]]
@A_L2L1_3_0_cons_buff_0 = external global [16 x [32 x i16]]
@A_L3L2_0_cons_buff_1 = external global [1024 x i16]
@A_L3L2_0_cons_buff_0 = external global [1024 x i16]
@A_L3L2_1_cons_buff_1 = external global [1024 x i16]
@A_L3L2_1_cons_buff_0 = external global [1024 x i16]
@B_L3L2_0_cons_buff_1 = external global [1536 x i16]
@B_L3L2_0_cons_buff_0 = external global [1536 x i16]
@B_L2L1_0_3_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_0_3_cons_buff_0 = external global [32 x [48 x i16]]
@B_L2L1_0_2_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_0_2_cons_buff_0 = external global [32 x [48 x i16]]
@B_L2L1_0_1_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_0_1_cons_buff_0 = external global [32 x [48 x i16]]
@B_L2L1_0_0_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_0_0_cons_buff_0 = external global [32 x [48 x i16]]
@B_L3L2_1_cons_buff_1 = external global [1536 x i16]
@B_L3L2_1_cons_buff_0 = external global [1536 x i16]
@B_L2L1_1_3_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_1_3_cons_buff_0 = external global [32 x [48 x i16]]
@B_L2L1_1_2_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_1_2_cons_buff_0 = external global [32 x [48 x i16]]
@B_L2L1_1_1_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_1_1_cons_buff_0 = external global [32 x [48 x i16]]
@B_L2L1_1_0_cons_buff_1 = external global [32 x [48 x i16]]
@B_L2L1_1_0_cons_buff_0 = external global [32 x [48 x i16]]
@C_L1L2_0_0_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_0_0_buff_0 = external global [16 x [48 x i32]]
@C_L1L2_0_1_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_0_1_buff_0 = external global [16 x [48 x i32]]
@C_L1L2_0_2_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_0_2_buff_0 = external global [16 x [48 x i32]]
@C_L1L2_0_3_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_0_3_buff_0 = external global [16 x [48 x i32]]
@C_L2L3_0_buff_1 = external global [3072 x i32]
@C_L2L3_0_buff_0 = external global [3072 x i32]
@C_L1L2_1_0_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_1_0_buff_0 = external global [16 x [48 x i32]]
@C_L1L2_1_1_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_1_1_buff_0 = external global [16 x [48 x i32]]
@C_L1L2_1_2_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_1_2_buff_0 = external global [16 x [48 x i32]]
@C_L1L2_1_3_buff_1 = external global [16 x [48 x i32]]
@C_L1L2_1_3_buff_0 = external global [16 x [48 x i32]]
@C_L2L3_1_buff_1 = external global [3072 x i32]
@C_L2L3_1_buff_0 = external global [3072 x i32]
@C_L2L3_1_cons = external global [3072 x i32]
@C_L2L3_1 = external global [3072 x i32]
@C_L1L2_1_3_cons = external global [16 x [48 x i32]]
@C_L1L2_1_3 = external global [16 x [48 x i32]]
@C_L1L2_1_2_cons = external global [16 x [48 x i32]]
@C_L1L2_1_2 = external global [16 x [48 x i32]]
@C_L1L2_1_1_cons = external global [16 x [48 x i32]]
@C_L1L2_1_1 = external global [16 x [48 x i32]]
@C_L1L2_1_0_cons = external global [16 x [48 x i32]]
@C_L1L2_1_0 = external global [16 x [48 x i32]]
@C_L2L3_0_cons = external global [3072 x i32]
@C_L2L3_0 = external global [3072 x i32]
@C_L1L2_0_3_cons = external global [16 x [48 x i32]]
@C_L1L2_0_3 = external global [16 x [48 x i32]]
@C_L1L2_0_2_cons = external global [16 x [48 x i32]]
@C_L1L2_0_2 = external global [16 x [48 x i32]]
@C_L1L2_0_1_cons = external global [16 x [48 x i32]]
@C_L1L2_0_1 = external global [16 x [48 x i32]]
@C_L1L2_0_0_cons = external global [16 x [48 x i32]]
@C_L1L2_0_0 = external global [16 x [48 x i32]]
@B_L2L1_1_0_cons = external global [32 x [48 x i16]]
@B_L2L1_1_1_cons = external global [32 x [48 x i16]]
@B_L2L1_1_2_cons = external global [32 x [48 x i16]]
@B_L2L1_1_3_cons = external global [32 x [48 x i16]]
@B_L2L1_1 = external global [32 x [48 x i16]]
@B_L3L2_1_cons = external global [1536 x i16]
@B_L3L2_1 = external global [1536 x i16]
@B_L2L1_0_0_cons = external global [32 x [48 x i16]]
@B_L2L1_0_1_cons = external global [32 x [48 x i16]]
@B_L2L1_0_2_cons = external global [32 x [48 x i16]]
@B_L2L1_0_3_cons = external global [32 x [48 x i16]]
@B_L2L1_0 = external global [32 x [48 x i16]]
@B_L3L2_0_cons = external global [1536 x i16]
@B_L3L2_0 = external global [1536 x i16]
@A_L3L2_1_cons = external global [1024 x i16]
@A_L3L2_1 = external global [1024 x i16]
@A_L3L2_0_cons = external global [1024 x i16]
@A_L3L2_0 = external global [1024 x i16]
@A_L2L1_3_0_cons = external global [16 x [32 x i16]]
@A_L2L1_3_1_cons = external global [16 x [32 x i16]]
@A_L2L1_3 = external global [16 x [32 x i16]]
@A_L2L1_2_0_cons = external global [16 x [32 x i16]]
@A_L2L1_2_1_cons = external global [16 x [32 x i16]]
@A_L2L1_2 = external global [16 x [32 x i16]]
@A_L2L1_1_0_cons = external global [16 x [32 x i16]]
@A_L2L1_1_1_cons = external global [16 x [32 x i16]]
@A_L2L1_1 = external global [16 x [32 x i16]]
@A_L2L1_0_0_cons = external global [16 x [32 x i16]]
@A_L2L1_0_1_cons = external global [16 x [32 x i16]]
@A_L2L1_0 = external global [16 x [32 x i16]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @zero_scalar_i32(ptr)

declare void @zero_i32(ptr)

declare void @matmul_scalar_i16_i32(ptr, ptr, ptr)

declare void @matmul_i16_i32(ptr, ptr, ptr)

define void @core_1_5() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_1_3_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_1_3_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_1_3_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_3_1_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_3_1_cons_buff_0, ptr @B_L2L1_1_3_cons_buff_0, ptr @C_L1L2_1_3_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_1_3_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_3_1_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_3_1_cons_buff_1, ptr @B_L2L1_1_3_cons_buff_1, ptr @C_L1L2_1_3_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_1_3_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_1_3_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_1_3_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_3_1_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_3_1_cons_buff_0, ptr @B_L2L1_1_3_cons_buff_0, ptr @C_L1L2_1_3_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_1_3_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_3_1_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_3_1_cons_buff_1, ptr @B_L2L1_1_3_cons_buff_1, ptr @C_L1L2_1_3_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

define void @core_0_5() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_0_3_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_0_3_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_0_3_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_3_0_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_3_0_cons_buff_0, ptr @B_L2L1_0_3_cons_buff_0, ptr @C_L1L2_0_3_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_0_3_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_3_0_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_3_0_cons_buff_1, ptr @B_L2L1_0_3_cons_buff_1, ptr @C_L1L2_0_3_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_0_3_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_0_3_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_0_3_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_3_0_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_3_0_cons_buff_0, ptr @B_L2L1_0_3_cons_buff_0, ptr @C_L1L2_0_3_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_0_3_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_3_0_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_3_0_cons_buff_1, ptr @B_L2L1_0_3_cons_buff_1, ptr @C_L1L2_0_3_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

define void @core_1_4() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_1_2_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_1_2_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_1_2_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_2_1_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_2_1_cons_buff_0, ptr @B_L2L1_1_2_cons_buff_0, ptr @C_L1L2_1_2_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_1_2_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_2_1_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_2_1_cons_buff_1, ptr @B_L2L1_1_2_cons_buff_1, ptr @C_L1L2_1_2_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_1_2_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_1_2_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_1_2_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_2_1_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_2_1_cons_buff_0, ptr @B_L2L1_1_2_cons_buff_0, ptr @C_L1L2_1_2_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_1_2_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_2_1_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_2_1_cons_buff_1, ptr @B_L2L1_1_2_cons_buff_1, ptr @C_L1L2_1_2_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

define void @core_0_4() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_0_2_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_0_2_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_0_2_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_2_0_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_2_0_cons_buff_0, ptr @B_L2L1_0_2_cons_buff_0, ptr @C_L1L2_0_2_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_0_2_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_2_0_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_2_0_cons_buff_1, ptr @B_L2L1_0_2_cons_buff_1, ptr @C_L1L2_0_2_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_0_2_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_0_2_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_0_2_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_2_0_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_2_0_cons_buff_0, ptr @B_L2L1_0_2_cons_buff_0, ptr @C_L1L2_0_2_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_0_2_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_2_0_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_2_0_cons_buff_1, ptr @B_L2L1_0_2_cons_buff_1, ptr @C_L1L2_0_2_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

define void @core_1_3() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_1_1_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_1_1_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_1_1_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_1_1_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_1_1_cons_buff_0, ptr @B_L2L1_1_1_cons_buff_0, ptr @C_L1L2_1_1_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_1_1_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_1_1_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_1_1_cons_buff_1, ptr @B_L2L1_1_1_cons_buff_1, ptr @C_L1L2_1_1_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_1_1_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_1_1_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_1_1_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_1_1_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_1_1_cons_buff_0, ptr @B_L2L1_1_1_cons_buff_0, ptr @C_L1L2_1_1_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_1_1_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_1_1_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_1_1_cons_buff_1, ptr @B_L2L1_1_1_cons_buff_1, ptr @C_L1L2_1_1_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

define void @core_0_3() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_0_1_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_0_1_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_0_1_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_1_0_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_1_0_cons_buff_0, ptr @B_L2L1_0_1_cons_buff_0, ptr @C_L1L2_0_1_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_0_1_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_1_0_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_1_0_cons_buff_1, ptr @B_L2L1_0_1_cons_buff_1, ptr @C_L1L2_0_1_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_0_1_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_0_1_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_0_1_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_1_0_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_1_0_cons_buff_0, ptr @B_L2L1_0_1_cons_buff_0, ptr @C_L1L2_0_1_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_0_1_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_1_0_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_1_0_cons_buff_1, ptr @B_L2L1_0_1_cons_buff_1, ptr @C_L1L2_0_1_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

define void @core_1_2() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_1_0_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_1_0_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_1_0_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_0_1_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_0_1_cons_buff_0, ptr @B_L2L1_1_0_cons_buff_0, ptr @C_L1L2_1_0_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_1_0_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_0_1_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_0_1_cons_buff_1, ptr @B_L2L1_1_0_cons_buff_1, ptr @C_L1L2_1_0_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_1_0_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_1_0_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_1_0_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_0_1_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_0_1_cons_buff_0, ptr @B_L2L1_1_0_cons_buff_0, ptr @C_L1L2_1_0_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_1_0_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_0_1_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_0_1_cons_buff_1, ptr @B_L2L1_1_0_cons_buff_1, ptr @C_L1L2_1_0_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %41, %0
  %2 = phi i64 [ %42, %41 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %43

4:                                                ; preds = %39, %1
  %5 = phi i64 [ %40, %39 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 80
  br i1 %6, label %7, label %41

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %8 = and i64 ptrtoint (ptr @C_L1L2_0_0_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  call void @zero_i32(ptr @C_L1L2_0_0_buff_0)
  br label %10

10:                                               ; preds = %13, %7
  %11 = phi i64 [ %22, %13 ], [ 0, %7 ]
  %12 = icmp slt i64 %11, 28
  br i1 %12, label %13, label %23

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %14 = and i64 ptrtoint (ptr @B_L2L1_0_0_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @A_L2L1_0_0_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @matmul_i16_i32(ptr @A_L2L1_0_0_cons_buff_0, ptr @B_L2L1_0_0_cons_buff_0, ptr @C_L1L2_0_0_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %9)
  %18 = and i64 ptrtoint (ptr @B_L2L1_0_0_cons_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @A_L2L1_0_0_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @matmul_i16_i32(ptr @A_L2L1_0_0_cons_buff_1, ptr @B_L2L1_0_0_cons_buff_1, ptr @C_L1L2_0_0_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %22 = add i64 %11, 2
  br label %10

23:                                               ; preds = %10
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @C_L1L2_0_0_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  call void @zero_i32(ptr @C_L1L2_0_0_buff_1)
  br label %26

26:                                               ; preds = %29, %23
  %27 = phi i64 [ %38, %29 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 28
  br i1 %28, label %29, label %39

29:                                               ; preds = %26
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %30 = and i64 ptrtoint (ptr @B_L2L1_0_0_cons_buff_0 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = and i64 ptrtoint (ptr @A_L2L1_0_0_cons_buff_0 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @matmul_i16_i32(ptr @A_L2L1_0_0_cons_buff_0, ptr @B_L2L1_0_0_cons_buff_0, ptr @C_L1L2_0_0_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %25)
  %34 = and i64 ptrtoint (ptr @B_L2L1_0_0_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = and i64 ptrtoint (ptr @A_L2L1_0_0_cons_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @matmul_i16_i32(ptr @A_L2L1_0_0_cons_buff_1, ptr @B_L2L1_0_0_cons_buff_1, ptr @C_L1L2_0_0_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %38 = add i64 %27, 2
  br label %26

39:                                               ; preds = %26
  call void @llvm.aie2.release(i32 53, i32 1)
  %40 = add i64 %5, 2
  br label %4

41:                                               ; preds = %4
  %42 = add i64 %2, 1
  br label %1

43:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
