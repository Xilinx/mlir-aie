; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@shim_to_mem_cons_buff_1 = external global [8 x [20 x i32]]
@shim_to_mem_cons_buff_0 = external global [8 x [20 x i32]]
@mem_to_comp_cons_buff_1 = external global [8 x [20 x i32]]
@mem_to_comp_cons_buff_0 = external global [8 x [20 x i32]]
@comp_to_mem_buff_1 = external global [8 x [20 x i32]]
@comp_to_mem_buff_0 = external global [8 x [20 x i32]]
@comp_to_mem_cons_buff_1 = external global [8 x [20 x i32]]
@comp_to_mem_cons_buff_0 = external global [8 x [20 x i32]]
@mem_to_shim_cons = external global [8 x [20 x i32]]
@mem_to_shim = external global [8 x [20 x i32]]
@comp_to_mem_cons = external global [8 x [20 x i32]]
@comp_to_mem = external global [8 x [20 x i32]]
@mem_to_comp_cons = external global [8 x [20 x i32]]
@mem_to_comp = external global [8 x [20 x i32]]
@shim_to_mem_cons = external global [8 x [20 x i32]]
@shim_to_mem = external global [8 x [20 x i32]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %36, %0
  %2 = phi i64 [ %37, %36 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %38

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %18, %4
  %6 = phi i64 [ %19, %18 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %20

8:                                                ; preds = %11, %5
  %9 = phi i64 [ %17, %11 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 20
  br i1 %10, label %11, label %18

11:                                               ; preds = %8
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  %12 = mul i64 %6, 20
  %13 = add i64 %12, %9
  %14 = getelementptr i32, ptr @mem_to_comp_cons_buff_0, i64 %13
  %15 = load i32, ptr %14, align 4
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  %16 = getelementptr i32, ptr @comp_to_mem_buff_0, i64 %13
  store i32 %15, ptr %16, align 4
  %17 = add i64 %9, 1
  br label %8

18:                                               ; preds = %8
  %19 = add i64 %6, 1
  br label %5

20:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %21

21:                                               ; preds = %34, %20
  %22 = phi i64 [ %35, %34 ], [ 0, %20 ]
  %23 = icmp slt i64 %22, 8
  br i1 %23, label %24, label %36

24:                                               ; preds = %27, %21
  %25 = phi i64 [ %33, %27 ], [ 0, %21 ]
  %26 = icmp slt i64 %25, 20
  br i1 %26, label %27, label %34

27:                                               ; preds = %24
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_1, i64 32) ]
  %28 = mul i64 %22, 20
  %29 = add i64 %28, %25
  %30 = getelementptr i32, ptr @mem_to_comp_cons_buff_1, i64 %29
  %31 = load i32, ptr %30, align 4
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_1, i64 32) ]
  %32 = getelementptr i32, ptr @comp_to_mem_buff_1, i64 %29
  store i32 %31, ptr %32, align 4
  %33 = add i64 %25, 1
  br label %24

34:                                               ; preds = %24
  %35 = add i64 %22, 1
  br label %21

36:                                               ; preds = %21
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %37 = add i64 %2, 2
  br label %1

38:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %39

39:                                               ; preds = %52, %38
  %40 = phi i64 [ %53, %52 ], [ 0, %38 ]
  %41 = icmp slt i64 %40, 8
  br i1 %41, label %42, label %54

42:                                               ; preds = %45, %39
  %43 = phi i64 [ %51, %45 ], [ 0, %39 ]
  %44 = icmp slt i64 %43, 20
  br i1 %44, label %45, label %52

45:                                               ; preds = %42
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  %46 = mul i64 %40, 20
  %47 = add i64 %46, %43
  %48 = getelementptr i32, ptr @mem_to_comp_cons_buff_0, i64 %47
  %49 = load i32, ptr %48, align 4
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  %50 = getelementptr i32, ptr @comp_to_mem_buff_0, i64 %47
  store i32 %49, ptr %50, align 4
  %51 = add i64 %43, 1
  br label %42

52:                                               ; preds = %42
  %53 = add i64 %40, 1
  br label %39

54:                                               ; preds = %39
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
