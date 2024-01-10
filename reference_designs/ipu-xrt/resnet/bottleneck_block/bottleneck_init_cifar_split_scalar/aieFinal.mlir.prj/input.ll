; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@rtp5 = external global [16 x i32]
@rtp4 = external global [16 x i32]
@rtp3 = external global [16 x i32]
@rtp2 = external global [16 x i32]
@inOF_act_L3L2_1_cons_buff_3 = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2_1_cons_buff_2 = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2_1_cons_buff_1 = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2_1_cons_buff_0 = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2_0_cons_buff_1 = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2_0_cons_buff_0 = external global [32 x [1 x [64 x i8]]]
@skip_buf_cons_buff_1 = external global [32 x [1 x [64 x i8]]]
@skip_buf_cons_buff_0 = external global [32 x [1 x [64 x i8]]]
@inOF_wts_0_L3L2_cons_buff_0 = external global [73728 x i8]
@wts_buf_00_cons_buff_0 = external global [4096 x i8]
@wts_buf_01_1_cons_buff_0 = external global [36864 x i8]
@wts_buf_01_0_cons_buff_0 = external global [36864 x i8]
@wts_buf_02_cons_buff_0 = external global [32768 x i8]
@act_2_3_4_buff_1 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_buff_0 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_1_cons_buff_3 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_1_cons_buff_2 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_1_cons_buff_1 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_1_cons_buff_0 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_0_cons_buff_3 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_0_cons_buff_2 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_0_cons_buff_1 = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_0_cons_buff_0 = external global [32 x [1 x [64 x i8]]]
@act_3_5_buff_1 = external global [32 x [1 x [32 x i8]]]
@act_3_5_buff_0 = external global [32 x [1 x [32 x i8]]]
@act_4_5_buff_1 = external global [32 x [1 x [32 x i8]]]
@act_4_5_buff_0 = external global [32 x [1 x [32 x i8]]]
@outOFL2L3_buff_1 = external global [32 x [1 x [256 x i8]]]
@outOFL2L3_buff_0 = external global [32 x [1 x [256 x i8]]]
@outOFL2L3_cons = external global [32 x [1 x [256 x i8]]]
@outOFL2L3 = external global [32 x [1 x [256 x i8]]]
@act_4_5 = external global [32 x [1 x [32 x i8]]]
@act_3_5 = external global [32 x [1 x [32 x i8]]]
@act_2_3_4_0_cons = external global [32 x [1 x [64 x i8]]]
@act_2_3_4_1_cons = external global [32 x [1 x [64 x i8]]]
@act_2_3_4 = external global [32 x [1 x [64 x i8]]]
@wts_buf_02_cons = external global [32768 x i8]
@wts_buf_02 = external global [32768 x i8]
@wts_buf_01_0_cons = external global [36864 x i8]
@wts_buf_01_1_cons = external global [36864 x i8]
@wts_buf_01 = external global [36864 x i8]
@wts_buf_00_cons = external global [4096 x i8]
@wts_buf_00 = external global [4096 x i8]
@inOF_wts_0_L3L2_cons = external global [73728 x i8]
@inOF_wts_0_L3L2 = external global [73728 x i8]
@skip_buf_cons = external global [32 x [1 x [64 x i8]]]
@skip_buf = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2_0_cons = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2_1_cons = external global [32 x [1 x [64 x i8]]]
@inOF_act_L3L2 = external global [32 x [1 x [64 x i8]]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @conv2dk1(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @conv2dk3(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @conv2dk1_skip(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32)

define void @sequence(ptr %0, ptr %1, ptr %2) {
  ret void
}

define void @core_0_4() {
  br label %1

1:                                                ; preds = %33, %0
  %2 = phi i64 [ %34, %33 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %35

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  %5 = and i64 ptrtoint (ptr @rtp5 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = load i32, ptr @rtp5, align 4
  call void @llvm.assume(i1 %6)
  %8 = load i32, ptr getelementptr (i32, ptr @rtp5, i32 1), align 4
  call void @llvm.assume(i1 %6)
  %9 = load i32, ptr getelementptr (i32, ptr @rtp5, i32 2), align 4
  br label %10

10:                                               ; preds = %13, %4
  %11 = phi i64 [ %32, %13 ], [ 0, %4 ]
  %12 = icmp slt i64 %11, 32
  br i1 %12, label %13, label %33

13:                                               ; preds = %10
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 37, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %14 = and i64 ptrtoint (ptr @outOFL2L3_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @act_4_5_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  %18 = and i64 ptrtoint (ptr @act_3_5_buff_0 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  %20 = and i64 ptrtoint (ptr @wts_buf_02_cons_buff_0 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  %22 = and i64 ptrtoint (ptr @skip_buf_cons_buff_0 to i64), 31
  %23 = icmp eq i64 %22, 0
  call void @llvm.assume(i1 %23)
  call void @conv2dk1_skip(ptr @act_3_5_buff_0, ptr @act_4_5_buff_0, ptr @wts_buf_02_cons_buff_0, ptr @outOFL2L3_buff_0, ptr @skip_buf_cons_buff_0, i32 32, i32 64, i32 256, i32 64, i32 %7, i32 %8, i32 %9)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 36, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 37, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %24 = and i64 ptrtoint (ptr @outOFL2L3_buff_1 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  %26 = and i64 ptrtoint (ptr @act_4_5_buff_1 to i64), 31
  %27 = icmp eq i64 %26, 0
  call void @llvm.assume(i1 %27)
  %28 = and i64 ptrtoint (ptr @act_3_5_buff_1 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  call void @llvm.assume(i1 %21)
  %30 = and i64 ptrtoint (ptr @skip_buf_cons_buff_1 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  call void @conv2dk1_skip(ptr @act_3_5_buff_1, ptr @act_4_5_buff_1, ptr @wts_buf_02_cons_buff_0, ptr @outOFL2L3_buff_1, ptr @skip_buf_cons_buff_1, i32 32, i32 64, i32 256, i32 64, i32 %7, i32 %8, i32 %9)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 36, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  %32 = add i64 %11, 2
  br label %10

33:                                               ; preds = %10
  call void @llvm.aie2.release(i32 50, i32 1)
  %34 = add i64 %2, 1
  br label %1

35:                                               ; preds = %1
  ret void
}

define void @core_0_5() {
  br label %1

1:                                                ; preds = %48, %0
  %2 = phi i64 [ %49, %48 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967292
  br i1 %3, label %4, label %50

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %5 = and i64 ptrtoint (ptr @act_4_5_buff_0 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_0 to i64), 31
  %8 = icmp eq i64 %7, 0
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  %9 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_1 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @wts_buf_01_1_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %13

13:                                               ; preds = %16, %4
  %14 = phi i64 [ %23, %16 ], [ 0, %4 ]
  %15 = icmp slt i64 %14, 28
  br i1 %15, label %16, label %24

16:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %17 = and i64 ptrtoint (ptr @act_4_5_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  %19 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_2 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %20)
  %21 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_3 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %23 = add i64 %14, 4
  br label %13

24:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @act_4_5_buff_1 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  %27 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_2 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  %29 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_3 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %31

31:                                               ; preds = %34, %24
  %32 = phi i64 [ %35, %34 ], [ 0, %24 ]
  %33 = icmp slt i64 %32, 28
  br i1 %33, label %34, label %36

34:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %35 = add i64 %32, 4
  br label %31

36:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %41, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 28
  br i1 %39, label %40, label %42

40:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %41 = add i64 %38, 4
  br label %37

42:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %43

43:                                               ; preds = %46, %42
  %44 = phi i64 [ %47, %46 ], [ 0, %42 ]
  %45 = icmp slt i64 %44, 28
  br i1 %45, label %46, label %48

46:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %47 = add i64 %44, 4
  br label %43

48:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  %49 = add i64 %2, 4
  br label %1

50:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %51 = and i64 ptrtoint (ptr @act_4_5_buff_0 to i64), 31
  %52 = icmp eq i64 %51, 0
  call void @llvm.assume(i1 %52)
  %53 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_0 to i64), 31
  %54 = icmp eq i64 %53, 0
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  %55 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_1 to i64), 31
  %56 = icmp eq i64 %55, 0
  call void @llvm.assume(i1 %56)
  %57 = and i64 ptrtoint (ptr @wts_buf_01_1_cons_buff_0 to i64), 31
  %58 = icmp eq i64 %57, 0
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %59

59:                                               ; preds = %62, %50
  %60 = phi i64 [ %69, %62 ], [ 0, %50 ]
  %61 = icmp slt i64 %60, 28
  br i1 %61, label %62, label %70

62:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %63 = and i64 ptrtoint (ptr @act_4_5_buff_1 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  %65 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_2 to i64), 31
  %66 = icmp eq i64 %65, 0
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %66)
  %67 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_3 to i64), 31
  %68 = icmp eq i64 %67, 0
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %69 = add i64 %60, 4
  br label %59

70:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %71 = and i64 ptrtoint (ptr @act_4_5_buff_1 to i64), 31
  %72 = icmp eq i64 %71, 0
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  %73 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_2 to i64), 31
  %74 = icmp eq i64 %73, 0
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  %75 = and i64 ptrtoint (ptr @act_2_3_4_1_cons_buff_3 to i64), 31
  %76 = icmp eq i64 %75, 0
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %77

77:                                               ; preds = %80, %70
  %78 = phi i64 [ %81, %80 ], [ 0, %70 ]
  %79 = icmp slt i64 %78, 28
  br i1 %79, label %80, label %82

80:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %81 = add i64 %78, 4
  br label %77

82:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %83

83:                                               ; preds = %86, %82
  %84 = phi i64 [ %87, %86 ], [ 0, %82 ]
  %85 = icmp slt i64 %84, 28
  br i1 %85, label %86, label %88

86:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %87 = add i64 %84, 4
  br label %83

88:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_0, ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_1, ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_1_cons_buff_2, ptr @act_2_3_4_1_cons_buff_3, ptr @act_2_3_4_1_cons_buff_3, ptr @wts_buf_01_1_cons_buff_0, ptr @act_4_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 32)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_0_3() {
  br label %1

1:                                                ; preds = %48, %0
  %2 = phi i64 [ %49, %48 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967292
  br i1 %3, label %4, label %50

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %5 = and i64 ptrtoint (ptr @act_3_5_buff_0 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_0 to i64), 31
  %8 = icmp eq i64 %7, 0
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  %9 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_1 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @wts_buf_01_0_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %13

13:                                               ; preds = %16, %4
  %14 = phi i64 [ %23, %16 ], [ 0, %4 ]
  %15 = icmp slt i64 %14, 28
  br i1 %15, label %16, label %24

16:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %17 = and i64 ptrtoint (ptr @act_3_5_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  %19 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_2 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %20)
  %21 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_3 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %23 = add i64 %14, 4
  br label %13

24:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @act_3_5_buff_1 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  %27 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_2 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  %29 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_3 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %31

31:                                               ; preds = %34, %24
  %32 = phi i64 [ %35, %34 ], [ 0, %24 ]
  %33 = icmp slt i64 %32, 28
  br i1 %33, label %34, label %36

34:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %35 = add i64 %32, 4
  br label %31

36:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %41, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 28
  br i1 %39, label %40, label %42

40:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %41 = add i64 %38, 4
  br label %37

42:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %43

43:                                               ; preds = %46, %42
  %44 = phi i64 [ %47, %46 ], [ 0, %42 ]
  %45 = icmp slt i64 %44, 28
  br i1 %45, label %46, label %48

46:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %47 = add i64 %44, 4
  br label %43

48:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  %49 = add i64 %2, 4
  br label %1

50:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %51 = and i64 ptrtoint (ptr @act_3_5_buff_0 to i64), 31
  %52 = icmp eq i64 %51, 0
  call void @llvm.assume(i1 %52)
  %53 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_0 to i64), 31
  %54 = icmp eq i64 %53, 0
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  %55 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_1 to i64), 31
  %56 = icmp eq i64 %55, 0
  call void @llvm.assume(i1 %56)
  %57 = and i64 ptrtoint (ptr @wts_buf_01_0_cons_buff_0 to i64), 31
  %58 = icmp eq i64 %57, 0
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %59

59:                                               ; preds = %62, %50
  %60 = phi i64 [ %69, %62 ], [ 0, %50 ]
  %61 = icmp slt i64 %60, 28
  br i1 %61, label %62, label %70

62:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %63 = and i64 ptrtoint (ptr @act_3_5_buff_1 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  %65 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_2 to i64), 31
  %66 = icmp eq i64 %65, 0
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %66)
  %67 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_3 to i64), 31
  %68 = icmp eq i64 %67, 0
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %69 = add i64 %60, 4
  br label %59

70:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %71 = and i64 ptrtoint (ptr @act_3_5_buff_1 to i64), 31
  %72 = icmp eq i64 %71, 0
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  %73 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_2 to i64), 31
  %74 = icmp eq i64 %73, 0
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  %75 = and i64 ptrtoint (ptr @act_2_3_4_0_cons_buff_3 to i64), 31
  %76 = icmp eq i64 %75, 0
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %77

77:                                               ; preds = %80, %70
  %78 = phi i64 [ %81, %80 ], [ 0, %70 ]
  %79 = icmp slt i64 %78, 28
  br i1 %79, label %80, label %82

80:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %81 = add i64 %78, 4
  br label %77

82:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 0, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %83

83:                                               ; preds = %86, %82
  %84 = phi i64 [ %87, %86 ], [ 0, %82 ]
  %85 = icmp slt i64 %84, 28
  br i1 %85, label %86, label %88

86:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %87 = add i64 %84, 4
  br label %83

88:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_0, ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_1, ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_0, i32 32, i32 64, i32 32, i32 3, i32 3, i32 1, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @conv2dk3(ptr @act_2_3_4_0_cons_buff_2, ptr @act_2_3_4_0_cons_buff_3, ptr @act_2_3_4_0_cons_buff_3, ptr @wts_buf_01_0_cons_buff_0, ptr @act_3_5_buff_1, i32 32, i32 64, i32 32, i32 3, i32 3, i32 2, i32 11, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %23, %0
  %2 = phi i64 [ %24, %23 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %25

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  %5 = and i64 ptrtoint (ptr @rtp2 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = load i32, ptr @rtp2, align 4
  br label %8

8:                                                ; preds = %11, %4
  %9 = phi i64 [ %22, %11 ], [ 0, %4 ]
  %10 = icmp slt i64 %9, 32
  br i1 %10, label %11, label %23

11:                                               ; preds = %8
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %12 = and i64 ptrtoint (ptr @act_2_3_4_buff_0 to i64), 31
  %13 = icmp eq i64 %12, 0
  call void @llvm.assume(i1 %13)
  %14 = and i64 ptrtoint (ptr @wts_buf_00_cons_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @conv2dk1(ptr @inOF_act_L3L2_0_cons_buff_0, ptr @wts_buf_00_cons_buff_0, ptr @act_2_3_4_buff_0, i32 32, i32 64, i32 64, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %18 = and i64 ptrtoint (ptr @act_2_3_4_buff_1 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  call void @llvm.assume(i1 %15)
  %20 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_1 to i64), 31
  %21 = icmp eq i64 %20, 0
  call void @llvm.assume(i1 %21)
  call void @conv2dk1(ptr @inOF_act_L3L2_0_cons_buff_1, ptr @wts_buf_00_cons_buff_0, ptr @act_2_3_4_buff_1, i32 32, i32 64, i32 64, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %22 = add i64 %9, 2
  br label %8

23:                                               ; preds = %8
  call void @llvm.aie2.release(i32 50, i32 1)
  %24 = add i64 %2, 1
  br label %1

25:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
