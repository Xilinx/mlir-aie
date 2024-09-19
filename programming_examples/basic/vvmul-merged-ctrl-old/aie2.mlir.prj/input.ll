; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@objFifo_out0_buff_1 = external global [64 x [64 x i8]]
@objFifo_out0_buff_0 = external global [64 x [64 x i8]]
@objFifo_in0_cons_buff_1 = external global [64 x [64 x i8]]
@objFifo_in0_cons_buff_0 = external global [64 x [64 x i8]]
@objFifo_out1_buff_1 = external global [64 x [64 x i8]]
@objFifo_out1_buff_0 = external global [64 x [64 x i8]]
@objFifo_in1_cons_buff_1 = external global [64 x [64 x i8]]
@objFifo_in1_cons_buff_0 = external global [64 x [64 x i8]]
@objFifo_in0 = external global [56 x [56 x i8]]
@objFifo_out0 = external global [64 x [64 x i8]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @core_0_2() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %1

1:                                                ; preds = %19, %0
  %2 = phi i64 [ %20, %19 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 64
  br i1 %3, label %4, label %21

4:                                                ; preds = %7, %1
  %5 = phi i64 [ %18, %7 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 64
  br i1 %6, label %7, label %19

7:                                                ; preds = %4
  %8 = and i64 ptrtoint (ptr @objFifo_in1_cons_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  %10 = mul i64 %2, 64
  %11 = add i64 %10, %5
  %12 = getelementptr i8, ptr @objFifo_in1_cons_buff_0, i64 %11
  %13 = load i8, ptr %12, align 1
  %14 = add i8 %13, 12
  %15 = and i64 ptrtoint (ptr @objFifo_out1_buff_0 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = getelementptr i8, ptr @objFifo_out1_buff_0, i64 %11
  store i8 %14, ptr %17, align 1
  %18 = add i64 %5, 1
  br label %4

19:                                               ; preds = %4
  %20 = add i64 %2, 1
  br label %1

21:                                               ; preds = %1
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
