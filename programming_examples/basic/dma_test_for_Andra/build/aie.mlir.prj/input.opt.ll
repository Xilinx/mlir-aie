; ModuleID = '/scratch/endrtaka/mlir_aie_dir/mlir-aie/programming_examples/basic/dma_test_for_Andra/build/aie.mlir.prj/input.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@mem_to_comp_cons_buff_1 = external global [8 x [20 x i32]]
@mem_to_comp_cons_buff_0 = external global [8 x [20 x i32]]
@comp_to_mem_buff_1 = external global [8 x [20 x i32]]
@comp_to_mem_buff_0 = external global [8 x [20 x i32]]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

; Function Attrs: nounwind
define void @core_0_2() local_unnamed_addr #0 {
.new:
  br label %0

0:                                                ; preds = %0, %.new
  %niter = phi i64 [ 0, %.new ], [ %niter.next.3, %0 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_0, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_0, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_1, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_1, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_0, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_0, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_1, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_1, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_0, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_0, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_1, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_1, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_0, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_0, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_1, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_1, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %niter.next.3 = add nuw nsw i64 %niter, 4
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, 4611686018427387900
  br i1 %niter.ncmp.3, label %.epil.preheader, label %0

.epil.preheader:                                  ; preds = %0, %.epil.preheader
  %epil.iter = phi i64 [ %epil.iter.next, %.epil.preheader ], [ 0, %0 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_0, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_0, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_1, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_1, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, 3
  br i1 %epil.iter.cmp.not, label %.epilog-lcssa, label %.epil.preheader, !llvm.loop !1

.epilog-lcssa:                                    ; preds = %.epil.preheader
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mem_to_comp_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @comp_to_mem_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(640) @comp_to_mem_buff_0, ptr noundef nonnull align 32 dereferenceable(640) @mem_to_comp_cons_buff_0, i20 640, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i20(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i20, i1 immarg) #2

attributes #0 = { nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.disable"}
