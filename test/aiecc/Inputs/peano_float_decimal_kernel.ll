; Copyright (C) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; A merge-mode ("inline") kernel artifact carrying narrow float constants in
; every position downgradeIRForPeano has to reason about. Owned by
; peano_compat_float_decimal_merge.mlir.
;
; The merge linker reprints this module with aiecc's LLVM, which spells a
; float/half constant as a short decimal whenever that decimal round-trips in
; the narrow type -- a spelling Peano's older parser rejects. The constants are
; written here in the hex form so the intent is fixed whatever a given LLVM
; prints; the reprint is what introduces the decimals.
;
; The (ptr, ptr) signature is aiecc's bare-pointer memref calling convention.

target triple = "aie2"

; Typed position: a constant array, the shape the LUT in a real attention
; kernel takes. 2.5 is exact as a double and must survive as printed.
@lut_f = linkonce_odr global [4 x float] [float 0x400921FA00000000,
                                          float 0x3FBC28F5C0000000,
                                          float 0xBF6A8292A0000000,
                                          float 2.500000e+00], align 4
; Same, for half, which takes its own 16-bit hex form.
@lut_h = linkonce_odr global [2 x half] [half 0xH2F0A, half 0xH4100], align 2

define linkonce_odr void @merge_kernel(ptr %in, ptr %out) #0 {
entry:
  %x = load float, ptr %in, align 4
  %l = load float, ptr @lut_f, align 4
  %s = fadd float %x, %l
  ; Bare operand position: the constant carries no type keyword of its own and
  ; takes float from the instruction.
  %r = fmul float %s, 0x3FBC28F5C0000000
  ; Mixed-type line: the double must keep its decimal spelling, which Peano
  ; accepts, rather than be rewritten under the float semantics named later on
  ; the same line.
  %d = fptrunc double 1.100000e-01 to float
  %rd = fadd float %r, %d
  %h = load half, ptr @lut_h, align 2
  %hf = fpext half %h to float
  %acc = fadd float %rd, %hf
  store float %acc, ptr %out, align 4
  ret void
}

attributes #0 = { alwaysinline }
