-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Exports
import IronVerify.Schema.Proofs.Domain
import IronVerify.Schema.Instances.npu2_i16_i16_64x64x32_3colCheck
import IronVerify.Schema.Instances.npu2_i16_i16_64x64x32_8colCheck

/-! Built by `check.py --steps instances`, after `instances.py` generates `Schema/Instances`. -/

namespace IronVerify.Schema.WholeArray

open IronVerify

/-! ### Two real IRON exports

whole_array.py's defaults (i16, m = 64, k = 64, n = 32) at 3 columns (the uneven A split) and 8
columns, exported from the live objects and checked against the schema in `Schema/Instances`. -/

namespace Export

theorem valid3 : Instances.npu2_i16_i16_64x64x32_3colCheck.p.Valid := by
  constructor <;> decide

theorem valid8 : Instances.npu2_i16_i16_64x64x32_8colCheck.p.Valid := by
  constructor <;> decide

/-- Every core of the 3-column export runs a matmul whose exported contract (output `i16`, no
saturation set by it or by any kernel on its core) gives the value domain `wrap 16`. -/
theorem domain_3col :
    ∀ e ∈ Instances.npu2_i16_i16_64x64x32_3col.kernels, accDomain? e.2 = some (.wrap 16) := by
  decide

theorem domain_8col :
    ∀ e ∈ Instances.npu2_i16_i16_64x64x32_8col.kernels, accDomain? e.2 = some (.wrap 16) := by
  decide

/-- whole_array at 3 columns computes `C = A @ B` as its kernels' contracts say it stores it: for
every `M`, `K`, `N` its guards accept, the exported contracts give every core one wrap width
`w`, and wherever one of the three drains writes `C[i, j]`, the exact dot product read at that
width is the drained value. (`domain_3col`: `w = 16`.) -/
theorem matmul_3col (hA hB : Nat → ℤ) {M K N : Int}
    (hok : (Instances.npu2_i16_i16_64x64x32_3col.design.seq (M, K, N)).ok = true)
    {i j c q : Nat} (hiM : i < M.toNat) (hjN : j < N.toNat) (hc : c < 3)
    (hq : (Instances.npu2_i16_i16_64x64x32_3col.design.host (M, K, N) (cL3 c))[q]? =
      some (2, ((i * N.toNat + j : Nat) : Int))) :
    ∃ w, (∀ e ∈ Instances.npu2_i16_i16_64x64x32_3col.kernels, accDomain? e.2 = some (.wrap w)) ∧
      ((ValueDomain.read (.wrap w)
          (∑ x ∈ Finset.range K.toNat, hA (i * K.toNat + x) * hB (x * N.toNat + j)) : ℤ) : Wrap w) =
        drainVal Instances.npu2_i16_i16_64x64x32_3col.design
          Instances.npu2_i16_i16_64x64x32_3colCheck.p
          (fun x => (hA x : Wrap w)) (fun x => (hB x : Wrap w)) (M, K, N)
          Instances.npu2_i16_i16_64x64x32_3colCheck.p.matmul (fun _ _ => 0) c q :=
  ⟨16, domain_3col,
    tMM_read Instances.npu2_i16_i16_64x64x32_3colCheck.inst valid3 16 hA hB hok hiM hjN hc hq⟩

/-- The same at 8 columns: the C write is unique across its eight drains (T-C), and its value is
the dot product at the width every core's exported contract gives (`domain_8col`: 16). -/
theorem matmul_8col (hA hB : Nat → ℤ) {M K N : Int}
    (hok : (Instances.npu2_i16_i16_64x64x32_8col.design.seq (M, K, N)).ok = true)
    {i j c q : Nat} (hiM : i < M.toNat) (hjN : j < N.toNat) (hc : c < 8)
    (hq : (Instances.npu2_i16_i16_64x64x32_8col.design.host (M, K, N) (cL3 c))[q]? =
      some (2, ((i * N.toNat + j : Nat) : Int))) :
    ∃ w, (∀ e ∈ Instances.npu2_i16_i16_64x64x32_8col.kernels, accDomain? e.2 = some (.wrap w)) ∧
      ((ValueDomain.read (.wrap w)
          (∑ x ∈ Finset.range K.toNat, hA (i * K.toNat + x) * hB (x * N.toNat + j)) : ℤ) : Wrap w) =
        drainVal Instances.npu2_i16_i16_64x64x32_8col.design
          Instances.npu2_i16_i16_64x64x32_8colCheck.p
          (fun x => (hA x : Wrap w)) (fun x => (hB x : Wrap w)) (M, K, N)
          Instances.npu2_i16_i16_64x64x32_8colCheck.p.matmul (fun _ _ => 0) c q :=
  ⟨16, domain_8col,
    tMM_read Instances.npu2_i16_i16_64x64x32_8colCheck.inst valid8 16 hA hB hok hiM hjN hc hq⟩

end Export

end IronVerify.Schema.WholeArray
