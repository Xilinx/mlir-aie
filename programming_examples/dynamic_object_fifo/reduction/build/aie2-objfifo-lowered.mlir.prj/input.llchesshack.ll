; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@input_fifo_cons_buff_3 = external global [10 x i32]
@input_fifo_cons_buff_2 = external global [10 x i32]
@input_fifo_cons_buff_1 = external global [10 x i32]
@input_fifo_cons_buff_0 = external global [10 x i32]
@output_fifo_buff_1 = external global [10 x i32]
@output_fifo_buff_0 = external global [10 x i32]
@output_fifo_cons = external global [10 x i32]
@output_fifo = external global [10 x i32]
@input_fifo_cons = external global [10 x i32]
@input_fifo = external global [10 x i32]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @sum_10_i32(ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %33, %0
  %2 = phi i64 [ %40, %33 ], [ 0, %0 ]
  %3 = phi i64 [ %38, %33 ], [ 0, %0 ]
  %4 = phi i64 [ %39, %33 ], [ 0, %0 ]
  %5 = icmp slt i64 %2, 10
  br i1 %5, label %6, label %41

6:                                                ; preds = %1
  %7 = srem i64 %3, 2
  %8 = srem i64 %4, 2
  %9 = trunc i64 %7 to i32
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %13
  ]

10:                                               ; preds = %6, %6
  %11 = and i64 ptrtoint (ptr @input_fifo_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  br label %16

13:                                               ; preds = %6
  %14 = and i64 ptrtoint (ptr @input_fifo_cons_buff_2 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  br label %16

16:                                               ; preds = %10, %13
  %17 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_2, i64 0, [1 x i64] [i64 10], [1 x i64] [i64 1] }, %13 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_0, i64 0, [1 x i64] [i64 10], [1 x i64] [i64 1] }, %10 ]
  switch i32 %9, label %18 [
    i32 0, label %18
    i32 1, label %21
  ]

18:                                               ; preds = %16, %16
  %19 = and i64 ptrtoint (ptr @input_fifo_cons_buff_1 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  br label %24

21:                                               ; preds = %16
  %22 = and i64 ptrtoint (ptr @input_fifo_cons_buff_3 to i64), 31
  %23 = icmp eq i64 %22, 0
  call void @llvm.assume(i1 %23)
  br label %24

24:                                               ; preds = %18, %21
  %25 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_3, i64 0, [1 x i64] [i64 10], [1 x i64] [i64 1] }, %21 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_1, i64 0, [1 x i64] [i64 10], [1 x i64] [i64 1] }, %18 ]
  %26 = trunc i64 %8 to i32
  switch i32 %26, label %30 [
    i32 0, label %27
    i32 1, label %30
  ]

27:                                               ; preds = %24
  %28 = and i64 ptrtoint (ptr @output_fifo_buff_0 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  br label %33

30:                                               ; preds = %24, %24
  %31 = and i64 ptrtoint (ptr @output_fifo_buff_1 to i64), 31
  %32 = icmp eq i64 %31, 0
  call void @llvm.assume(i1 %32)
  br label %33

33:                                               ; preds = %30, %27
  %34 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @output_fifo_buff_1, i64 0, [1 x i64] [i64 10], [1 x i64] [i64 1] }, %30 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @output_fifo_buff_0, i64 0, [1 x i64] [i64 10], [1 x i64] [i64 1] }, %27 ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 49, i32 -2)
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 1
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 1
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 1
  call void @sum_10_i32(ptr %35, ptr %36, ptr %37)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 48, i32 2)
  %38 = add i64 %3, 1
  %39 = add i64 %4, 1
  %40 = add i64 %2, 1
  br label %1

41:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
