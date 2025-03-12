; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@inA_cons_buff_1 = external global [64 x [120 x i8]]
@inA_cons_buff_0 = external global [64 x [120 x i8]]
@memA_cons_buff_1 = external global [64 x [120 x i8]]
@memA_cons_buff_0 = external global [64 x [120 x i8]]
@inB_cons_buff_1 = external global [120 x [96 x i8]]
@inB_cons_buff_0 = external global [120 x [96 x i8]]
@memB_cons_buff_1 = external global [120 x [96 x i8]]
@memB_cons_buff_0 = external global [120 x [96 x i8]]
@memC_buff_1 = external global [64 x [96 x i16]]
@memC_buff_0 = external global [64 x [96 x i16]]
@memC_cons_buff_1 = external global [64 x [96 x i16]]
@memC_cons_buff_0 = external global [64 x [96 x i16]]
@outC_cons = external global [64 x [96 x i16]]
@outC = external global [64 x [96 x i16]]
@memC_cons = external global [64 x [96 x i16]]
@memC = external global [64 x [96 x i16]]
@memB_cons = external global [120 x [96 x i8]]
@memB = external global [120 x [96 x i8]]
@inB_cons = external global [120 x [96 x i8]]
@inB = external global [120 x [96 x i8]]
@memA_cons = external global [64 x [120 x i8]]
@memA = external global [64 x [120 x i8]]
@inA_cons = external global [64 x [120 x i8]]
@inA = external global [64 x [120 x i8]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @zero_i16(ptr)

declare void @matmul_i8_i16(ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %21, %0
  %2 = phi i64 [ %22, %21 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %23

4:                                                ; preds = %19, %1
  %5 = phi i64 [ %20, %19 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 40
  br i1 %6, label %7, label %21

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @memC_buff_0, i64 32) ]
  call void @zero_i16(ptr @memC_buff_0)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %12, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 4
  br i1 %10, label %11, label %13

11:                                               ; preds = %8
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @memC_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memA_cons_buff_0, i64 32) ]
  call void @matmul_i8_i16(ptr @memA_cons_buff_0, ptr @memB_cons_buff_0, ptr @memC_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @memC_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memA_cons_buff_1, i64 32) ]
  call void @matmul_i8_i16(ptr @memA_cons_buff_1, ptr @memB_cons_buff_1, ptr @memC_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %12 = add i64 %9, 2
  br label %8

13:                                               ; preds = %8
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @memC_buff_1, i64 32) ]
  call void @zero_i16(ptr @memC_buff_1)
  br label %14

14:                                               ; preds = %17, %13
  %15 = phi i64 [ %18, %17 ], [ 0, %13 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %19

17:                                               ; preds = %14
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @memC_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memA_cons_buff_0, i64 32) ]
  call void @matmul_i8_i16(ptr @memA_cons_buff_0, ptr @memB_cons_buff_0, ptr @memC_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @memC_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memA_cons_buff_1, i64 32) ]
  call void @matmul_i8_i16(ptr @memA_cons_buff_1, ptr @memB_cons_buff_1, ptr @memC_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %18 = add i64 %15, 2
  br label %14

19:                                               ; preds = %14
  call void @llvm.aie2.release(i32 53, i32 1)
  %20 = add i64 %5, 2
  br label %4

21:                                               ; preds = %4
  %22 = add i64 %2, 1
  br label %1

23:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
