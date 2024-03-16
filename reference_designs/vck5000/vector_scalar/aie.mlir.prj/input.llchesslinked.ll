; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target triple = "aie"

%struct.ipd.custom_type.uint1_t.uint1_t = type { i1 }

@in_cons_buff_1 = external global [16 x i32]
@in_cons_buff_0 = external global [16 x i32]
@out_buff_1 = external global [16 x i32]
@out_buff_0 = external global [16 x i32]

define void @sequence(ptr %0, ptr %1, ptr %2) {
  ret void
}

define void @core_6_2() {
  br label %1

1:                                                ; preds = %37, %0
  %2 = phi i64 [ %38, %37 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %39

4:                                                ; preds = %35, %1
  %5 = phi i64 [ %36, %35 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4
  br i1 %6, label %7, label %37

7:                                                ; preds = %4
  call void @llvm.aie.lock.acquire.reg(i32 16, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 18, i32 0)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %20, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 16
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = and i64 ptrtoint (ptr @in_cons_buff_0 to i64), 31
  %13 = icmp eq i64 %12, 0
  call void @llvm.assume(i1 %13)
  %14 = getelementptr i32, ptr @in_cons_buff_0, i64 %9
  %15 = load i32, ptr %14, align 4
  %16 = mul i32 %15, 3
  %17 = and i64 ptrtoint (ptr @out_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  %19 = getelementptr i32, ptr @out_buff_0, i64 %9
  store i32 %16, ptr %19, align 4
  %20 = add i64 %9, 1
  br label %8

21:                                               ; preds = %8
  call void @llvm.aie.lock.release.reg(i32 16, i32 0)
  call void @llvm.aie.lock.release.reg(i32 18, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 17, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 19, i32 0)
  br label %22

22:                                               ; preds = %25, %21
  %23 = phi i64 [ %34, %25 ], [ 0, %21 ]
  %24 = icmp slt i64 %23, 16
  br i1 %24, label %25, label %35

25:                                               ; preds = %22
  %26 = and i64 ptrtoint (ptr @in_cons_buff_1 to i64), 31
  %27 = icmp eq i64 %26, 0
  call void @llvm.assume(i1 %27)
  %28 = getelementptr i32, ptr @in_cons_buff_1, i64 %23
  %29 = load i32, ptr %28, align 4
  %30 = mul i32 %29, 3
  %31 = and i64 ptrtoint (ptr @out_buff_1 to i64), 31
  %32 = icmp eq i64 %31, 0
  call void @llvm.assume(i1 %32)
  %33 = getelementptr i32, ptr @out_buff_1, i64 %23
  store i32 %30, ptr %33, align 4
  %34 = add i64 %23, 1
  br label %22

35:                                               ; preds = %22
  call void @llvm.aie.lock.release.reg(i32 17, i32 0)
  call void @llvm.aie.lock.release.reg(i32 19, i32 1)
  %36 = add i64 %5, 2
  br label %4

37:                                               ; preds = %4
  %38 = add i64 %2, 1
  br label %1

39:                                               ; preds = %1
  ret void
}

declare void @llvm.aie.lock.acquire.reg(i32, i32)

; Function Attrs: nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly
declare void @llvm.assume(i1 noundef) #0

declare void @llvm.aie.lock.release.reg(i32, i32)

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___acquire___reg(i32 %0, i32 %1) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i1 true }, i32 zeroext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #2

; Function Attrs: nounwind inaccessiblememonly
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___release___reg(i32 %0, i32 %1) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i1 true }, i32 zeroext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #3

attributes #0 = { nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly }
attributes #1 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { nounwind inaccessiblememonly "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind inaccessiblememonly "no-builtin-memcpy" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.linker.options = !{}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 15.0.5 (/u/sgasip/ipd/repositories/llvm_ipd 3a25925e0239306412dac02da5e4c8c51ae722e8)"}
