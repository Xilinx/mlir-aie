# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reproduce the IRON -> Lean proof of concept: exports, differential tests, proofs, schema
instances, mutations.

Run from an environment where `import aie` works (e.g. after `utils/env_setup.sh`)
and after `utils/iron_lean/setup.sh`. Each step prints PASS, FAIL or SKIPPED and
the script exits nonzero if any step fails.
"""

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "IronVerify"
STEPS = ["export", "difftest", "kernels", "sweep", "proofs", "instances", "mutations"]
PROOF_MODULES = (
    "IronVerify.Schema.Theorems",
    "IronVerify.Schema.Proofs.Kernel",
    "IronVerify.Schema.Exports",
)

# name: (n_aie_cols, MxKxN shapes); m, k, n stay at whole_array's defaults (64, 64, 32).
DESIGNS = {
    "WholeArrayC1": (
        1,
        ["256x128x64", "512x64x32", "768x192x96", "1280x64x64", "1792x64x32"],
    ),
    "WholeArrayC2": (2, ["256x64x64", "768x128x128", "1280x64x192"]),
    "WholeArrayC3": (3, ["256x64x96", "512x128x192", "768x192x288"]),
    "WholeArrayC4": (4, ["256x64x128", "512x128x256", "1280x64x128"]),
    "WholeArrayC5": (5, ["256x64x160", "512x128x320", "768x192x160"]),
    "WholeArrayC6": (6, ["256x64x192", "512x128x384", "768x192x192"]),
    "WholeArrayC7": (7, ["256x64x224", "512x128x448", "768x192x224"]),
    "WholeArrayC8": (8, ["256x64x256", "512x128x512", "1280x64x256"]),
}

# Lean 4 compiles `native_decide` to an auxiliary axiom, so allow-list rather than
# deny-list: a theorem may rest only on Lean's three standard axioms.
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
AXIOM_AUDIT = """{imports}
open Lean Elab Command

#eval show CommandElabM Unit from do
  for (n, ci) in (← getEnv).constants.toList do
    if (`IronVerify).isPrefixOf n then
      if let .thmInfo _ := ci then
        let axs ← liftCoreM (Lean.collectAxioms n)
        IO.println s!"AXIOMS {{n}} {{" ".intercalate (axs.toList.map toString)}}"
"""
GENERATED_THEOREM = re.compile(r"\.(eq_\d+|eq_def|_.*)$")


class Checker:
    """Runs the reproduction steps against the Lake project in `PROJECT`.

    Args:
        elan_home: Where `setup.sh` installed elan and the Lean toolchain.
        walk_count: How many random walks the walk differential test draws.
    """

    def __init__(self, elan_home: Path, walk_count: int):
        self.walk_count = walk_count
        self.env = dict(os.environ)
        self.env["ELAN_HOME"] = str(elan_home)
        self.env["PATH"] = f"{elan_home / 'bin'}{os.pathsep}{self.env['PATH']}"

    def run(self, cmd: list, cwd: Path = HERE, capture: bool = False):
        """Run `cmd`, echoing it; with `capture`, return its combined output too.

        Returns:
            The `subprocess.CompletedProcess`.
        """
        print("$", " ".join(str(c) for c in cmd), flush=True)
        return subprocess.run(
            [str(c) for c in cmd],
            cwd=cwd,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )

    def export(self) -> bool:
        """Re-export every design and compare it with the committed Lean file.

        Returns:
            Whether every committed export is up to date.
        """
        ok = True
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "IronVerify").mkdir()
            for name, (cols, _) in DESIGNS.items():
                cmd = [sys.executable, HERE / "export.py", "--lean-root", tmp]
                cmd += ["--n-aie-cols", cols, "--name", name]
                if self.run(cmd).returncode != 0:
                    ok = False
                    continue
                rel = Path("IronVerify/Designs") / f"{name}.lean"
                fresh = (Path(tmp) / rel).read_text().splitlines(keepends=True)
                committed = (PROJECT / rel).read_text().splitlines(keepends=True)
                if fresh != committed:
                    ok = False
                    print(f"STALE {rel}: re-run export.py --name {name}")
                    diff = difflib.unified_diff(committed, fresh, "committed", "fresh")
                    sys.stdout.writelines(list(diff)[:40])
        return ok

    def difftest(self) -> bool:
        """Compare Lean's evaluation with the numpy oracle: walks, then every design and shape.

        Returns:
            Whether every comparison agrees.
        """
        if self.run(["lake", "build", "ironeval"], cwd=PROJECT).returncode != 0:
            return False
        cmd = [sys.executable, HERE / "difftest_walk.py", "--count", self.walk_count]
        ok = self.run(cmd).returncode == 0
        for name, (cols, shapes) in DESIGNS.items():
            cmd = [sys.executable, HERE / "difftest_design.py", "--name", name]
            cmd += ["--n-aie-cols", cols, "--shapes", *shapes]
            ok &= self.run(cmd).returncode == 0
        return ok

    def proofs(self) -> bool:
        """Build the schema theorems and their proofs, then audit the axioms each theorem uses.

        Returns:
            Whether the proofs build without `sorry` and use only the standard axioms.
        """
        modules = list(PROOF_MODULES)
        build = self.run(["lake", "build", *modules], cwd=PROJECT, capture=True)
        print(build.stdout[-4000:])
        if build.returncode != 0 or "declaration uses `sorry`" in build.stdout:
            print("proof build failed or uses sorry")
            return False

        imports = "".join(f"import {m}\n" for m in modules)
        with tempfile.NamedTemporaryFile("w", suffix=".lean") as audit:
            audit.write(AXIOM_AUDIT.format(imports=imports))
            audit.flush()
            out = self.run(
                ["lake", "env", "lean", audit.name], cwd=PROJECT, capture=True
            )
        if out.returncode != 0:
            print(out.stdout[-4000:])
            return False

        ok, shown = True, 0
        for line in out.stdout.splitlines():
            if not line.startswith("AXIOMS "):
                continue
            _, name, *axioms = line.split()
            extra = set(axioms) - ALLOWED_AXIOMS
            if extra:
                ok = False
                print(f"DISALLOWED AXIOMS {name}: {sorted(extra)}")
            if name.startswith("IronVerify.Schema.") and not GENERATED_THEOREM.search(
                name
            ):
                shown += 1
                print(f"  {name}: {', '.join(axioms) or 'no axioms'}")
        print(f"{shown} theorems in IronVerify.Schema")
        return ok and shown > 0

    def instances(self) -> bool:
        """Generate the schema instances, build them and audit their axioms.

        Every shipped whole_array configuration is exported from its live objects into
        `Schema/Instances/` (generated, not committed); then each `Instance` proof (the export
        is `schema p`) and the worked examples are built and may rest only on the standard
        axioms.

        Returns:
            Whether every instance builds without `sorry` and uses allowed axioms.
        """
        if self.run(
            [sys.executable, HERE / "instances.py", "--mode", "write"]
        ).returncode:
            print("instance export failed")
            return False
        modules = ["IronVerify.Schema.Instances", "IronVerify.Schema.Examples"]
        build = self.run(["lake", "build", *modules], cwd=PROJECT, capture=True)
        print(build.stdout[-2000:])
        if build.returncode != 0 or "declaration uses `sorry`" in build.stdout:
            print("instance build failed or uses sorry")
            return False
        with tempfile.NamedTemporaryFile("w", suffix=".lean") as audit:
            audit.write(
                AXIOM_AUDIT.format(imports="".join(f"import {m}\n" for m in modules))
            )
            audit.flush()
            out = self.run(
                ["lake", "env", "lean", audit.name], cwd=PROJECT, capture=True
            )
        if out.returncode != 0:
            print(out.stdout[-4000:])
            return False
        ok, proved = True, 0
        for line in out.stdout.splitlines():
            if not line.startswith("AXIOMS "):
                continue
            _, name, *axioms = line.split()
            extra = set(axioms) - ALLOWED_AXIOMS
            if extra:
                ok = False
                print(f"DISALLOWED AXIOMS {name}: {sorted(extra)}")
            proved += name.startswith("IronVerify.Instances.") and name.endswith(
                ".inst"
            )
        print(f"{proved} exports proved to be schema instances")
        return ok and proved > 0

    def kernels(self) -> bool:
        """Check the exported kernel contracts against Python's own codec and reference.

        Returns:
            Whether every Lean operand layout matches `TensorLayout.encode` and every Lean
            matmul call matches `mm_tile_ref` in its value domain, on npu1 and npu2.
        """
        ok = True
        for script in ("difftest_layout.py", "difftest_matmul.py"):
            ok &= self.run([sys.executable, HERE / script]).returncode == 0
        return ok

    def sweep(self) -> bool:
        """Evaluate `schema p` against the numpy oracle on every shipped configuration.

        Returns:
            Whether the schema's provenance matches the live design's for every configuration.
        """
        cmd = [sys.executable, HERE / "schema_check.py", "--devices", "npu1", "npu2"]
        return self.run(cmd).returncode == 0

    def mutations(self, requested: bool) -> bool | None:
        """Run `mutations.py`, which expects every injected bug to be caught.

        Args:
            requested: Whether the user named this step explicitly, so its absence fails.

        Returns:
            Whether every mutation was caught, or None when skipped.
        """
        script = HERE / "mutations.py"
        if not script.exists():
            print("mutations.py not found")
            return False if requested else None
        return self.run([sys.executable, script]).returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--steps",
        nargs="+",
        choices=[*STEPS, "all"],
        default=["all"],
        help="steps to run, in order (default: all)",
    )
    p.add_argument(
        "--elan-home",
        type=Path,
        default=Path(os.environ.get("ELAN_HOME", PROJECT / ".elan")),
        help="elan install from setup.sh (default: $ELAN_HOME, else IronVerify/.elan)",
    )
    p.add_argument("--walk-count", type=int, default=2000)
    opts = p.parse_args()

    if not (opts.elan_home / "bin/lake").exists():
        p.error(f"no lake in {opts.elan_home}/bin: run setup.sh or pass --elan-home")
    steps = STEPS if "all" in opts.steps else opts.steps
    checker = Checker(opts.elan_home, opts.walk_count)
    results, seconds = {}, {}
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        start = time.monotonic()
        if step == "mutations":
            results[step] = checker.mutations(requested="all" not in opts.steps)
        else:
            results[step] = {
                "export": checker.export,
                "difftest": checker.difftest,
                "kernels": checker.kernels,
                "sweep": checker.sweep,
                "proofs": checker.proofs,
                "instances": checker.instances,
            }[step]()
        seconds[step] = time.monotonic() - start

    print("\n=== summary ===")
    for step, ok in results.items():
        status = "SKIPPED" if ok is None else "PASS" if ok else "FAIL"
        print(f"{status} {step} ({seconds[step]:.0f} s)")
    raise SystemExit(1 if any(ok is False for ok in results.values()) else 0)


if __name__ == "__main__":
    main()
