; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@objFifo_in0_cons_buff_1 = external global [16 x i32]
@objFifo_in0_cons_buff_0 = external global [16 x i32]
@objFifo_in1_cons_buff_1 = external global [8 x i32]
@objFifo_in1_cons_buff_0 = external global [8 x i32]
@objFifo_out1_buff_1 = external global [8 x i32]
@objFifo_out1_buff_0 = external global [8 x i32]
@objFifo_out0_buff_1 = external global [16 x i32]
@objFifo_out0_buff_0 = external global [16 x i32]
@objFifo_out0_cons = external global [16 x i32]
@objFifo_out0 = external global [16 x i32]
@objFifo_out1_cons = external global [8 x i32]
@objFifo_out1 = external global [8 x i32]
@objFifo_in1_cons = external global [8 x i32]
@objFifo_in1 = external global [8 x i32]
@objFifo_in0_cons = external global [16 x i32]
@objFifo_in0 = external global [16 x i32]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %32, %0
  %2 = phi i64 [ %33, %32 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 8
  br i1 %3, label %4, label %34

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %17, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  %9 = and i64 ptrtoint (ptr @objFifo_in1_cons_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = getelementptr i32, ptr @objFifo_in1_cons_buff_0, i64 %6
  %12 = load i32, ptr %11, align 4
  %13 = add i32 %12, 1
  %14 = and i64 ptrtoint (ptr @objFifo_out1_buff_0 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  %16 = getelementptr i32, ptr @objFifo_out1_buff_0, i64 %6
  store i32 %13, ptr %16, align 4
  %17 = add i64 %6, 1
  br label %5

18:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %19

19:                                               ; preds = %22, %18
  %20 = phi i64 [ %31, %22 ], [ 0, %18 ]
  %21 = icmp slt i64 %20, 8
  br i1 %21, label %22, label %32

22:                                               ; preds = %19
  %23 = and i64 ptrtoint (ptr @objFifo_in1_cons_buff_1 to i64), 31
  %24 = icmp eq i64 %23, 0
  call void @llvm.assume(i1 %24)
  %25 = getelementptr i32, ptr @objFifo_in1_cons_buff_1, i64 %20
  %26 = load i32, ptr %25, align 4
  %27 = add i32 %26, 1
  %28 = and i64 ptrtoint (ptr @objFifo_out1_buff_1 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  %30 = getelementptr i32, ptr @objFifo_out1_buff_1, i64 %20
  store i32 %27, ptr %30, align 4
  %31 = add i64 %20, 1
  br label %19

32:                                               ; preds = %19
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %33 = add i64 %2, 2
  br label %1

34:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
