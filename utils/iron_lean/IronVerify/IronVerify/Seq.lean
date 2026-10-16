-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Walk

namespace IronVerify

/-- A runtime sequence: shim transfers under the `scf.if`/`scf.for` structure that issued them. -/
inductive Seq where
  | xfer (chan : String) (buf : Nat) (w : Walk)
  | skip
  | seq (a b : Seq)
  | ite (c : Bool) (t e : Seq)
  | loop (lo hi : Int) (body : Int → Seq)
  | check (c : Bool)

namespace Seq

/-- The host elements the transfers on `ch` walk, in issue order. -/
def stream (ch : String) : Seq → List (Nat × Int)
  | xfer c b w => if c = ch then w.toList.map (b, ·) else []
  | skip => []
  | seq a b => stream ch a ++ stream ch b
  | ite c t e => if c then stream ch t else stream ch e
  | loop lo hi body => ((List.range (hi - lo).toNat).map fun i => stream ch (body (lo + i))).flatten
  | check _ => []

/-- Every `cf.assert` the sequence reaches holds. -/
def ok : Seq → Bool
  | xfer .. => true
  | skip => true
  | seq a b => ok a && ok b
  | ite c t e => if c then ok t else ok e
  | loop lo hi body => ((List.range (hi - lo).toNat).map fun i => ok (body (lo + i))).all id
  | check c => c

end Seq

end IronVerify
