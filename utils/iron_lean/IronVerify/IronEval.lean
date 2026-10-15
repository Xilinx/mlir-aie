-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Walk
import IronVerify.Seq
import IronVerify.Net
import IronVerify.Designs

open IronVerify

/-- Each stdin line is `offset s0 t0 s1 t1 ...`; prints the walk's element indices. -/
def parseWalk (line : String) : Option Walk := do
  let nums ← (line.splitOn " " |>.filter (· ≠ "")).mapM String.toInt?
  match nums with
  | [] => none
  | off :: rest =>
    let rec pairs : List Int → Option (List (Int × Int))
      | [] => some []
      | s :: t :: r => ((s, t) :: ·) <$> pairs r
      | [_] => none
    return ⟨off, ← pairs rest⟩

partial def walkLoop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  match parseWalk line.trimAsciiEnd.toString with
  | none => throw <| IO.userError s!"bad walk line: {line}"
  | some w => stdout.putStrLn (" ".intercalate (w.toList.map toString))
  stdout.flush
  walkLoop stdin stdout

def showSrc : Option Src → String
  | some (.host b i) => s!"h{b}:{i}"
  | some (.core w f j s) => s!"c{w}:{f}:{j}:{s}"
  | none => "none"

/-- Prints every object each worker receives and every host element each drain writes. -/
def evalDesign (net : Net) (seq : Seq) (stdout : IO.FS.Stream) : IO Unit := do
  let chans := net.fifos.map (·.name)
  let arrs := chans.map fun c => (c, (seq.stream c).toArray)
  let host : String → Array (Nat × Int) := fun c =>
    ((arrs.find? (·.1 = c)).map (·.2)).getD #[]
  stdout.putStrLn s!"ok {seq.ok}"
  for f in net.fifos do
    for (e, _) in f.cons do
      match e with
      | .worker w =>
        let some len := net.streamLen host f | continue
        for j in List.range (len / f.consSize) do
          let slots := (List.range f.consSize).map fun s => showSrc (net.deliver host f w j s)
          stdout.putStrLn s!"deliver {f.name} {w} {j} {" ".intercalate slots}"
      | .shim =>
        let arr := host f.name
        for p in List.range arr.size do
          let (b, i) := arr[p]!
          stdout.putStrLn s!"write {b} {i} {showSrc (net.stream host f p)}"
      | .link _ => pure ()

def main (args : List String) : IO UInt32 := do
  let stdout ← IO.getStdout
  match args with
  | ["walks"] => walkLoop (← IO.getStdin) stdout; return 0
  | "design" :: name :: params =>
    let some (_, net, seqOf) := Designs.registry.find? (·.1 = name)
      | throw <| IO.userError s!"unknown design {name}"
    let some seq := seqOf ((params.flatMap (·.splitOn "x")).filterMap String.toInt?)
      | throw <| IO.userError s!"bad parameters {params}"
    evalDesign net seq stdout
    return 0
  | _ => throw <| IO.userError "usage: ironeval walks | ironeval design NAME MxKxN (or M K N)"
