; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target triple = "aie"

%struct.ipd.custom_type.uint1_t.uint1_t = type { i8 }

@of_1_buff_0 = external global [256 x float]
@of_1_buff_1 = external global [256 x float]
@of_1_buff_2 = external global [256 x float]
@of_2_buff_0 = external global [256 x float]
@__chess_separator_dummy = external dso_local global i8*, align 4

define void @core_7_1() {
  call void @llvm.aie.lock.acquire.reg(i32 62, i32 0)
  call void @llvm.aie.lock.acquire.reg(i32 48, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 49, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 50, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 52, i32 0)
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_1_buff_0 to i64), i64 31), i64 0))
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_1_buff_1 to i64), i64 31), i64 0))
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_1_buff_2 to i64), i64 31), i64 0))
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_2_buff_0 to i64), i64 31), i64 0))
  call void @stencil_2d_7point_fp32(float* getelementptr inbounds ([256 x float], [256 x float]* @of_1_buff_0, i32 0, i32 0), float* getelementptr inbounds ([256 x float], [256 x float]* @of_1_buff_1, i32 0, i32 0), float* getelementptr inbounds ([256 x float], [256 x float]* @of_1_buff_2, i32 0, i32 0), float* getelementptr inbounds ([256 x float], [256 x float]* @of_2_buff_0, i32 0, i32 0))
  call void @llvm.aie.lock.release.reg(i32 48, i32 0)
  call void @llvm.aie.lock.release.reg(i32 52, i32 1)
  call void @llvm.aie.lock.release.reg(i32 62, i32 0)
  ret void
}

declare void @llvm.aie.lock.acquire.reg(i32, i32)

; Function Attrs:  nofree nosync nounwind willreturn inaccessiblememonly
declare void @llvm.assume(i1 ) #0

declare void @stencil_2d_7point_fp32(float*, float*, float*, float*)

declare void @llvm.aie.lock.release.reg(i32, i32)

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___acquire___reg(i32 , i32 ) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence() #4
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !3
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i8 -1 }, i32 zeroext %1) #5
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !3
  tail call addrspace(1) void @llvm.chess_memory_fence() #4
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #2

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind
define dso_local void @llvm___aie___lock___release___reg(i32 , i32 ) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence() #4
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !3
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext %0, %struct.ipd.custom_type.uint1_t.uint1_t { i8 -1 }, i32 zeroext %1) #5
  store volatile i8* inttoptr (i20 1 to i8*), i8** @__chess_separator_dummy, align 4, !tbaa !3
  tail call addrspace(1) void @llvm.chess_memory_fence() #4
  ret void
}

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release___uint_uint1_t___uint(i32 zeroext, %struct.ipd.custom_type.uint1_t.uint1_t, i32 zeroext) local_unnamed_addr addrspace(1) #3

attributes #0 = {  nofree nosync nounwind willreturn inaccessiblememonly }
attributes #1 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind willreturn }
attributes #3 = { nounwind inaccessiblememonly "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-builtin-memcpy" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { nounwind inaccessiblememonly "no-builtin-memcpy" }

!llvm.module.flags = !{!0, !1}
!llvm.linker.options = !{}
!llvm.ident = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 12.0.1 (sgasip@krachtcs10:ipd/repositories/llvm_ipd 91ba5f0998a58022b69d471ef7a8296d37aa98a9)"}
!3 = !{!4, !4, i64 0, i64 4}
!4 = !{!5, i64 4, !"__chess_separator_universe:any pointer"}
!5 = !{!6, i64 1, !"__chess_separator_universe:omnipotent char"}
!6 = !{!"Simple C++ TBAA"}
