-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Exports
import IronVerify.Schema.Proofs.Kernel

/-!
The value domain a core's exported kernel contracts give its accumulating kernel, and C = A @ B
read through that domain: the hardware stores the exact result narrowed as the domain says, and
for a `wrap w` domain that stored integer, taken in `ZMod (2 ^ w)`, is the export's drained value.
-/

namespace IronVerify

/-- The kernel on a core that accumulates into its output (the one with an `InOut` argument). -/
def accKernel? (onCore : List KernelD) : Option KernelD :=
  onCore.find? fun k => k.args.any (·.role == "InOut")

/-- The value domain the exported contracts give a core's accumulating kernel. -/
def accDomain? (onCore : List KernelD) : Option ValueDomain := do
  (← accKernel? onCore).domainOn onCore

namespace Schema.WholeArray.Export

variable {d : Design (Int × Int × Int)} {p : Params} (hi : Instance d p) (hp : p.Valid)
include hi hp

/-- **C = A @ B, as stored**: wherever a drain writes `C[i, j]`, the exact dot product read as a
`w`-bit two's-complement integer (`ValueDomain.read (.wrap w)`) is the export's drained value in
`ZMod (2 ^ w)`. -/
theorem tMM_read (w : Nat) (hA hB : Nat → ℤ) {M K N : Int} (hok : (d.seq (M, K, N)).ok = true)
    {i j c q : Nat} (hiM : i < M.toNat) (hjN : j < N.toNat) (hc : c < p.cols)
    (hq : (d.host (M, K, N) (cL3 c))[q]? = some (2, ((p.cIdx M.toNat N.toNat i j : Nat) : Int))) :
    ((ValueDomain.read (.wrap w)
        (∑ x ∈ Finset.range K.toNat, hA (p.aIdx K.toNat i x) * hB (p.bIdx K.toNat N.toNat x j)) :
        ℤ) : Wrap w) =
      drainVal d p (fun x => (hA x : Wrap w)) (fun x => (hB x : Wrap w)) (M, K, N) p.matmul
        (fun _ _ => 0) c q := by
  rw [ValueDomain.read_wrap, tMM_wrap hi hp w hA hB hok hiM hjN hc hq]

end Schema.WholeArray.Export

end IronVerify
