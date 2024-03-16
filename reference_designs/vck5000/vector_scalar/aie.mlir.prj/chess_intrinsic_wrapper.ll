; ModuleID = '/scratch/eddier/mlir-aie-2-22-2024/mlir-aie/install/aie_runtime_lib/AIE/chess_intrinsic_wrapper.cpp'
source_filename = "/scratch/eddier/mlir-aie-2-22-2024/mlir-aie/install/aie_runtime_lib/AIE/chess_intrinsic_wrapper.cpp"



%struct.ipd.custom_type.uint1_t.uint1_t = type { i1 }

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___acquire___reg(i32 %0, i32 %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i1 true }, i32 zeroext %1) #3
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___release___reg(i32 %0, i32 %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i1 true }, i32 zeroext %1) #3
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #1

; Function Attrs: inaccessiblememonly nounwind
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #2

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #2

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #2

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nounwind willreturn }
attributes #2 = { inaccessiblememonly nounwind "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { inaccessiblememonly nounwind "no-builtin-memcpy" }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 15.0.5 (/u/sgasip/ipd/repositories/llvm_ipd 3a25925e0239306412dac02da5e4c8c51ae722e8)"}
