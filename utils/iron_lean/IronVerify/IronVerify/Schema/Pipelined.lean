-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Lemmas

/-!
The shape `TaskGroup.pipelined(n, depth=4)` gives an exported runtime sequence: three guarded
peels, then a loop from 3. `peel3` folds it back into one loop over `n`.
-/

namespace IronVerify.Schema.WholeArray

theorem flatMap_cast (l : List Nat) :
    List.flatMap (fun a : Nat => [(Nat.cast a : Int)]) l = l.map (Nat.cast : Nat → Int) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.flatMap_cons, ih]; rfl

/-- `TaskGroup.pipelined(n, depth=4)`: three guarded peels, then a loop from 3. -/
theorem peel3 {α : Type} (g : Nat → List α) (n : Nat) :
    (if 0 < n then g 0 ++
      if 1 < n then g 1 ++
        if 2 < n then g 2 ++ ((List.range (n - 3)).map fun i => g (3 + i)).flatten
        else []
      else []
    else []) = ((List.range n).map g).flatten := by
  rcases n with _ | _ | _ | n
  · simp
  · simp
  · simp [List.range_succ]
  · rw [show n + 1 + 1 + 1 = 3 + n by omega, List.range_add]
    simp [List.range_succ, show 1 < 3 + n by omega, show 2 < 3 + n by omega, Function.comp_def]

end IronVerify.Schema.WholeArray
