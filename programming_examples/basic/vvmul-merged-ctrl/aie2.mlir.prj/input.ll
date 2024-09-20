; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@in1_cons_buff_1 = external global [16 x i32]
@in1_cons_buff_0 = external global [16 x i32]
@in2_cons_buff_1 = external global [16 x i32]
@in2_cons_buff_0 = external global [16 x i32]
@out_buff_1 = external global [16 x i32]
@out_buff_0 = external global [16 x i32]
@out_cons = external global [16 x i32]
@out = external global [16 x i32]
@in2_cons = external global [16 x i32]
@in2 = external global [16 x i32]
@in1_cons = external global [16 x i32]
@in1 = external global [16 x i32]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %45, %0
  %2 = phi i64 [ %46, %45 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %47

4:                                                ; preds = %43, %1
  %5 = phi i64 [ %44, %43 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 256
  br i1 %6, label %7, label %45

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %24, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 16
  br i1 %10, label %11, label %25

11:                                               ; preds = %8
  %12 = and i64 ptrtoint (ptr @in1_cons_buff_0 to i64), 31
  %13 = icmp eq i64 %12, 0
  call void @llvm.assume(i1 %13)
  %14 = getelementptr i32, ptr @in1_cons_buff_0, i64 %9
  %15 = load i32, ptr %14, align 4
  %16 = and i64 ptrtoint (ptr @in2_cons_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  %18 = getelementptr i32, ptr @in2_cons_buff_0, i64 %9
  %19 = load i32, ptr %18, align 4
  %20 = mul i32 %15, %19
  %21 = and i64 ptrtoint (ptr @out_buff_0 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  %23 = getelementptr i32, ptr @out_buff_0, i64 %9
  store i32 %20, ptr %23, align 4
  %24 = add i64 %9, 1
  br label %8

25:                                               ; preds = %8
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %26

26:                                               ; preds = %29, %25
  %27 = phi i64 [ %42, %29 ], [ 0, %25 ]
  %28 = icmp slt i64 %27, 16
  br i1 %28, label %29, label %43

29:                                               ; preds = %26
  %30 = and i64 ptrtoint (ptr @in1_cons_buff_1 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  %32 = getelementptr i32, ptr @in1_cons_buff_1, i64 %27
  %33 = load i32, ptr %32, align 4
  %34 = and i64 ptrtoint (ptr @in2_cons_buff_1 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  %36 = getelementptr i32, ptr @in2_cons_buff_1, i64 %27
  %37 = load i32, ptr %36, align 4
  %38 = mul i32 %33, %37
  %39 = and i64 ptrtoint (ptr @out_buff_1 to i64), 31
  %40 = icmp eq i64 %39, 0
  call void @llvm.assume(i1 %40)
  %41 = getelementptr i32, ptr @out_buff_1, i64 %27
  store i32 %38, ptr %41, align 4
  %42 = add i64 %27, 1
  br label %26

43:                                               ; preds = %26
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %44 = add i64 %5, 2
  br label %4

45:                                               ; preds = %4
  %46 = add i64 %2, 1
  br label %1

47:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
