; Copyright (C) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; A merge-mode ("inline") kernel artifact in the shape ExternalFunction(
; inline=True) hands aiecc: an `alwaysinline`, `linkonce_odr` definition in
; textual LLVM IR. Owned by peano_compat_merge_link.mlir.
;
; Hand-written rather than compiled so the constructs that broke the merge stay
; pinned whatever a given clang emits:
;
;   * generic intrinsics (llvm.fabs / fmuladd / smax / smin) -- their
;     declarations are what the merge linker decorates with attributes Peano
;     cannot parse. Real kernels reach them by canonicalization: a NaN check
;     becomes llvm.fabs, a multiply-add llvm.fmuladd, a clamp smax/smin.
;   * llvm.lifetime.start/end carrying the size operand Peano requires and a
;     newer LLVM drops.
;
; The (ptr, ptr) signature is aiecc's bare-pointer memref calling convention.

target triple = "aie2"

define linkonce_odr void @merge_kernel(ptr %in, ptr %out) #0 {
entry:
  %scratch = alloca [4 x float], align 4
  call void @llvm.lifetime.start.p0(i64 16, ptr %scratch)
  br label %body

body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %in.ptr = getelementptr inbounds i32, ptr %in, i32 %i
  %x = load i32, ptr %in.ptr, align 4
  ; A clamp to [0, 255]: llvm.smax / llvm.smin.
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 0)
  %clamped = call i32 @llvm.smin.i32(i32 %lo, i32 255)
  ; |v| * v + v staged through the local buffer: llvm.fabs / llvm.fmuladd.
  %v = sitofp i32 %clamped to float
  %mag = call float @llvm.fabs.f32(float %v)
  %slot = getelementptr inbounds [4 x float], ptr %scratch, i32 0, i32 0
  store float %mag, ptr %slot, align 4
  %mag.reloaded = load float, ptr %slot, align 4
  %acc = call float @llvm.fmuladd.f32(float %mag.reloaded, float %v, float %v)
  %res = fptosi float %acc to i32
  %out.ptr = getelementptr inbounds i32, ptr %out, i32 %i
  store i32 %res, ptr %out.ptr, align 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 16
  br i1 %done, label %exit, label %body

exit:
  call void @llvm.lifetime.end.p0(i64 16, ptr %scratch)
  ret void
}

declare float @llvm.fabs.f32(float)
declare float @llvm.fmuladd.f32(float, float, float)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

attributes #0 = { alwaysinline }
