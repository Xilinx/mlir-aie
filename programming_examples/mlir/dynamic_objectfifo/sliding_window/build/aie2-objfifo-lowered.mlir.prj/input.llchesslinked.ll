; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target triple = "aie2"

%struct.ipd.custom_type.uint2_t.uint2_t = type { i2 }

@input_fifo_cons_buff_2 = external global [64 x i32]
@input_fifo_cons_buff_1 = external global [64 x i32]
@input_fifo_cons_buff_0 = external global [64 x i32]
@output_fifo_buff_1 = external global [64 x i32]
@output_fifo_buff_0 = external global [64 x i32]

define void @sequence(ptr %0, ptr %1) {
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %54, %0
  %2 = phi i64 [ %56, %54 ], [ 0, %0 ]
  %3 = phi i64 [ %55, %54 ], [ 0, %0 ]
  %4 = icmp slt i64 %2, 4294967295
  br i1 %4, label %5, label %57

5:                                                ; preds = %1
  %6 = srem i64 %2, 2
  %7 = icmp eq i64 %6, 0
  %8 = srem i64 %3, 3
  %9 = trunc i64 %8 to i32
  switch i32 %9, label %10 [
    i32 0, label %10
    i32 1, label %13
    i32 2, label %16
  ]

10:                                               ; preds = %5, %5
  %11 = and i64 ptrtoint (ptr @input_fifo_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  br label %19

13:                                               ; preds = %5
  %14 = and i64 ptrtoint (ptr @input_fifo_cons_buff_1 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  br label %19

16:                                               ; preds = %5
  %17 = and i64 ptrtoint (ptr @input_fifo_cons_buff_2 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  br label %19

19:                                               ; preds = %16, %13, %10
  %20 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_2, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %16 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_1, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %13 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_0, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %10 ]
  switch i32 %9, label %27 [
    i32 0, label %21
    i32 1, label %24
    i32 2, label %27
  ]

21:                                               ; preds = %19
  %22 = and i64 ptrtoint (ptr @input_fifo_cons_buff_1 to i64), 31
  %23 = icmp eq i64 %22, 0
  call void @llvm.assume(i1 %23)
  br label %30

24:                                               ; preds = %19
  %25 = and i64 ptrtoint (ptr @input_fifo_cons_buff_2 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  br label %30

27:                                               ; preds = %19, %19
  %28 = and i64 ptrtoint (ptr @input_fifo_cons_buff_0 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  br label %30

30:                                               ; preds = %27, %24, %21
  %31 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_0, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %27 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_2, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %24 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @input_fifo_cons_buff_1, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %21 ]
  br i1 %7, label %32, label %35

32:                                               ; preds = %30
  %33 = and i64 ptrtoint (ptr @output_fifo_buff_0 to i64), 31
  %34 = icmp eq i64 %33, 0
  call void @llvm.assume(i1 %34)
  br label %38

35:                                               ; preds = %30
  %36 = and i64 ptrtoint (ptr @output_fifo_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  br label %38

38:                                               ; preds = %35, %32
  %39 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ { ptr inttoptr (i64 3735928559 to ptr), ptr @output_fifo_buff_1, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %35 ], [ { ptr inttoptr (i64 3735928559 to ptr), ptr @output_fifo_buff_0, i64 0, [1 x i64] [i64 64], [1 x i64] [i64 1] }, %32 ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  %40 = icmp eq i64 %2, 0
  %41 = icmp eq i64 %2, 4294967294
  br i1 %40, label %42, label %45

42:                                               ; preds = %38
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, 1
  call void @sum_64_i32(ptr %43, ptr %43, ptr %44)
  br label %54

45:                                               ; preds = %38
  br i1 %41, label %46, label %58

46:                                               ; preds = %58, %45
  %47 = phi i32 [ %59, %58 ], [ -2, %45 ]
  %48 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %60, %58 ], [ %31, %45 ]
  %49 = phi i32 [ %61, %58 ], [ 1, %45 ]
  %50 = phi i64 [ %62, %58 ], [ 1, %45 ]
  call void @llvm.aie2.acquire(i32 49, i32 %47)
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %52 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, 1
  %53 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, 1
  call void @sum_64_i32(ptr %51, ptr %52, ptr %53)
  call void @llvm.aie2.release(i32 48, i32 %49)
  br label %54

54:                                               ; preds = %46, %42
  %55 = phi i64 [ %50, %46 ], [ 0, %42 ]
  call void @llvm.aie2.release(i32 51, i32 1)
  %56 = add i64 %2, 1
  br label %1

57:                                               ; preds = %1
  ret void

58:                                               ; preds = %45
  %59 = phi i32 [ -1, %45 ]
  %60 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %20, %45 ]
  %61 = phi i32 [ 2, %45 ]
  %62 = phi i64 [ 2, %45 ]
  br label %46
}

; Function Attrs: nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly
declare void @llvm.assume(i1 noundef) #0

declare void @llvm.aie2.acquire(i32, i32)

declare void @sum_64_i32(ptr, ptr, ptr)

declare void @llvm.aie2.release(i32, i32)

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___acquire(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext %0, i32 zeroext %1) #5
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #2

; Function Attrs: nounwind inaccessiblememonly
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext, i32 zeroext) local_unnamed_addr addrspace(1) #3

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___release(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext %0, i32 signext %1) #5
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext, i32 signext) local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind
define dso_local void @llvm___aie___event0() local_unnamed_addr addrspace(1) #4 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t zeroinitializer) #5
  ret void
}

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t) local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind
define dso_local void @llvm___aie___event1() local_unnamed_addr addrspace(1) #4 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t { i2 1 }) #5
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly }
attributes #1 = { mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { nounwind inaccessiblememonly "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { nounwind inaccessiblememonly "no-builtin-memcpy" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.linker.options = !{}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 15.0.5 (/u/sgasip/ipd/repositories/llvm_ipd 3a25925e0239306412dac02da5e4c8c51ae722e8)"}
