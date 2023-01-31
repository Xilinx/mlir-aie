; ModuleID = '/pub/scratch/gagsingh/mlir-aie/install/bin/aiecc/../../runtime_lib/chess_intrinsic_wrapper.cpp'
source_filename = "/pub/scratch/gagsingh/mlir-aie/install/bin/aiecc/../../runtime_lib/chess_intrinsic_wrapper.cpp"



%struct.ipd.custom_type.uint1_t.uint1_t = type { i8 }

@__chess_separator_dummy = external dso_local global i8*, align 4

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___acquire___reg(i32 %0, i32 %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence() #3
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !2
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i8 -1 }, i32 zeroext %1) #4
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !2
  tail call addrspace(1) void @llvm.chess_memory_fence() #3
  ret void
}

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___release___reg(i32 %0, i32 %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence() #3
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !2
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i8 -1 }, i32 zeroext %1) #4
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !2
  tail call addrspace(1) void @llvm.chess_memory_fence() #3
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #1

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #2

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind willreturn }
attributes #2 = { inaccessiblememonly nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-builtin-memcpy" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { inaccessiblememonly nounwind "no-builtin-memcpy" }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (sgasip@krachtcs10:ipd/repositories/llvm_ipd 91ba5f0998a58022b69d471ef7a8296d37aa98a9)"}
!2 = !{!3, !3, i64 0, i64 4}
!3 = !{!4, i64 4, !"__chess_separator_universe:any pointer"}
!4 = !{!5, i64 1, !"__chess_separator_universe:omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
