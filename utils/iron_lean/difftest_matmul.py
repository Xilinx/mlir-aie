# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Differential test: Lean's per-call `matmul` and value domain against the kernel contract.

For each whole_array configuration, the matmul kernel's contract is exported from the live
worker (`export.py --contracts`). Lean evaluates one call, `p.matmul`, on buffers packed with
the contract's own `TensorLayout.encode`, and reads the result in the value domain it selects
from the exported contracts (`KernelD.domainOn`). numpy computes the same call as
`mm_tile_ref(A, B) + C` narrowed the way the hardware stores it. Integer inputs span the
whole input range, so the sums overflow the output type. A forced saturating read of every
integer case checks that path too.
"""

import argparse
import itertools
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from aie.iron import Worker, str_to_dtype
from aie.iron.kernel import Kernel
from aie.iron.kernels.linalg import mm_tile_ref

from export import WHOLE_ARRAY, DesignExporter, whole_array_program

LEAN_ROOT = Path(__file__).parent / "IronVerify"

CONFIGS = {
    "npu2": [
        ("i8", "i8", 0),
        ("i8", "i32", 0),
        ("i16", "i16", 0),
        ("i16", "i32", 0),
        ("bf16", "bf16", 0),
        ("bf16", "f32", 0),
        ("bf16", "bf16", 1),
    ],
    "npu1": [
        ("i8", "i8", 0),
        ("i8", "i32", 0),
        ("i16", "i16", 0),
        ("i16", "i32", 0),
        ("bf16", "bf16", 0),
        ("bf16", "f32", 0),
    ],
}
TILES = [(32, 32, 32), (64, 64, 32), (16, 32, 16)]
INT_RANGE = {"i8": (-128, 127), "i16": (-32768, 32767)}
OUT_BITS = {"i8": 8, "i16": 16, "i32": 32}

DRIVER = """import IronVerify.Kernel
import IronVerify.Schema.Spec

open IronVerify IronVerify.Schema.WholeArray

def cases : Array (Params × KernelD × List KernelD) := #[
{cases}]

def words (line : String) : List Int :=
  (line.trimAsciiEnd.toString.splitOn " " |>.filter (· ≠ "")).filterMap String.toInt?

def evalCase (p : Params) (kd : KernelD) (core : List KernelD) (data : List Int) : String :=
  let A := data.take (p.m * p.k)
  let B := (data.drop (p.m * p.k)).take (p.k * p.n)
  let C := data.drop (p.m * p.k + p.k * p.n)
  let fA := fun q => A.getD q 0
  let fB := fun q => B.getD q 0
  let fC := fun a b => C.getD (a * p.n + b) 0
  let exact := (List.range (p.m * p.n)).map fun e =>
    p.matmul (R := Int) fA fB fC (e / p.n) (e % p.n)
  let domain := kd.domainOn core
  let stored := match domain with
    | some d => exact.map d.read
    | none => []
  let sat := match kd.output?.bind (fun o => intWidth? o.dtype) with
    | some (w, signed) => exact.map (ValueDomain.ofInt w signed .saturate).read
    | none => []
  let lay := fun (i : Nat) (f : Nat → Nat → Nat) (rows cols : Nat) =>
    match (kd.args[i]?).bind (·.layout) with
    | some L => (List.range (rows * cols)).all fun e =>
        L.slotElem (f (e / cols) (e % cols)) == some e
    | none => false
  let layouts := lay 0 p.layoutA p.m p.k && lay 1 p.layoutB p.k p.n && lay 2 p.layoutC p.m p.n
  s!"{reprStr domain}|{kd.accCovers}|{layouts}|{" ".intercalate (stored.map toString)}|{" ".intercalate (sat.map toString)}"

partial def loop (stdin stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  match words line with
  | i :: data =>
    match cases[i.toNat]? with
    | some (p, kd, core) => stdout.putStrLn (evalCase p kd core data)
    | none => throw <| IO.userError s!"no case {i}"
  | [] => pure ()
  stdout.flush
  loop stdin stdout

def main : IO Unit := do
  loop (← IO.getStdin) (← IO.getStdout)
"""


def wrap(v: np.ndarray, bits: int) -> np.ndarray:
    half = 1 << (bits - 1)
    return ((v + half) % (1 << bits)) - half


def matmul_kernel(worker: Worker) -> Kernel:
    return next(
        f
        for f in worker.flat_fn_args
        if isinstance(f, Kernel) and f.contract is not None and f.contract.reduction
    )


def sample(rng: np.random.Generator, dtype: str, shape: tuple) -> np.ndarray:
    if dtype in INT_RANGE:
        lo, hi = INT_RANGE[dtype]
        return rng.integers(lo, hi + 1, shape, dtype=np.int64)
    # Small integers keep the float sums exact, so Lean's exact result can be compared.
    return rng.integers(-8, 9, shape, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--devices", nargs="+", choices=["npu1", "npu2"], default=["npu1", "npu2"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tiles",
        nargs="+",
        choices=["x".join(map(str, t)) for t in TILES],
        default=["x".join(map(str, t)) for t in TILES],
    )
    opts = p.parse_args()
    rng = np.random.default_rng(opts.seed)

    cases, lines, expected, labels = [], [], [], []
    for device in opts.devices:
        for (din, dout, emu), (m, k, n), bc, cc in itertools.product(
            CONFIGS[device],
            [tuple(int(x) for x in t.split("x")) for t in opts.tiles],
            (0, 1),
            (0, 1),
        ):
            try:
                program = whole_array_program(
                    WHOLE_ARRAY, None, m, k, n, 1, device, din, dout, bc, cc, emu
                )
            except (ValueError, AssertionError):
                continue
            exporter = DesignExporter(program, canonical=True, contracts=True)
            worker = exporter.workers[0]
            fn = matmul_kernel(worker)
            r, s, t = fn.mac_dims
            core = [
                exporter.kernel(f) for f in worker.flat_fn_args if isinstance(f, Kernel)
            ]
            params = (
                f"{{ rows := 4, cols := 1, m := {m}, k := {k}, n := {n}, r := {r}, "
                f"s := {s}, t := {t}, bColMaj := {str(bool(bc)).lower()}, "
                f"cColMaj := {str(bool(cc)).lower()} }}"
            )
            cases.append(f"  ({params}, {exporter.kernel(fn)}, [{', '.join(core)}])")

            A = sample(rng, din, (m, k))
            B = sample(rng, din, (k, n))
            C0 = sample(rng, dout if dout in INT_RANGE else din, (m, n))
            la, lb, _ = fn.contract.layouts
            bufA = la.encode(A.astype(np.float64)).reshape(-1).astype(np.int64)
            bufB = lb.encode(B.astype(np.float64)).reshape(-1).astype(np.int64)
            ref = mm_tile_ref(A, B, dim_m=m, dim_k=k, dim_n=n).reshape(m, n) + C0
            if dout in OUT_BITS:
                bits = OUT_BITS[dout]
                stored = wrap(ref, bits)
                half = 1 << (bits - 1)
                sat = np.clip(ref, -half, half - 1)
            else:
                stored, sat = ref, None
            data = np.concatenate([bufA, bufB, C0.reshape(-1)])
            lines.append(f"{len(cases) - 1} " + " ".join(map(str, data.tolist())))
            expected.append(
                (
                    stored.reshape(-1).tolist(),
                    None if sat is None else sat.reshape(-1).tolist(),
                    dout,
                )
            )
            labels.append(
                f"{device} {din}-{dout}{' emu' if emu else ''} {m}x{k}x{n} b{bc}c{cc}"
            )

    with tempfile.TemporaryDirectory() as tmp:
        driver = Path(tmp) / "MatmulDriver.lean"
        driver.write_text(DRIVER.replace("{cases}", ",\n".join(cases)))
        out = subprocess.run(
            ["lake", "env", "lean", "--run", str(driver)],
            cwd=LEAN_ROOT,
            input="\n".join(lines) + "\n",
            capture_output=True,
            text=True,
        )
    if out.returncode:
        raise SystemExit(
            f"Lean driver failed:\n{out.stdout[-3000:]}{out.stderr[-3000:]}"
        )
    results = out.stdout.splitlines()
    assert len(results) == len(labels), (len(results), len(labels), out.stderr[-2000:])

    bad = 0
    rows = {}
    for label, line, (stored, sat, dout) in zip(labels, results, expected):
        domain, covers, layouts, got, got_sat = line.split("|")
        got = [int(x) for x in got.split()]
        got_sat = [int(x) for x in got_sat.split()]
        ok = layouts == "true" and covers == "true" and got == stored
        if sat is not None:
            ok = ok and got_sat == sat
        bad += not ok
        key = " ".join(label.split()[:2])
        rows.setdefault(key, [0, 0, domain])
        rows[key][0] += ok
        rows[key][1] += 1
        if not ok:
            print(
                f"FAIL {label}: domain={domain} accCovers={covers} layouts={layouts} "
                f"stored {'ok' if got == stored else 'differs'} "
                f"saturating {'ok' if sat is None or got_sat == sat else 'differs'}"
            )
    for key, (n_ok, n, domain) in rows.items():
        print(f"{key:24s} {n_ok}/{n}  {domain}")
    print(f"{len(labels) - bad}/{len(labels)} matmul calls agree")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
