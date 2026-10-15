# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Mutation suite: inject real bugs into whole_array and check each layer of the verifier sees them.

Each mutant is a copy of whole_array.py with one textual change. For the
unmutated design and for every mutant this runs:

* the design difftest (Lean evaluation of the symbolic export vs the numpy
  oracle). It checks the *model*, so a mutant is expected to PASS here too:
  both sides describe the same, buggy, design;
* the concrete matmul check (oracle data movement + kernel contract vs
  ``A @ B``), which must FAIL for a mutant;
* the proof check: the WholeArrayC1 proofs retargeted at the mutant's export,
  which must no longer build.

Mutant Lean files live in a scratch copy of the Lake project (`--work-dir`)
whose `.lake/packages` links to the in-tree one, so the in-tree project and
its design registry are untouched.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np

from difftest_design import compare
from aie.iron.device import from_name

from export import LEAN_ROOT, WHOLE_ARRAY, whole_array_program, write_design
from matmul_check import MatmulCheck

MUTATIONS = {
    "a_permute": (
        "wrong A permute: the repeat dim stays outermost, so A[row] selects a repeat",
        ".repeat(N // n // n_aie_cols)\n        .permute((1, 0, 2, 3, 4))",
        ".repeat(N // n // n_aie_cols)\n        .permute((0, 1, 2, 3, 4))",
    ),
    "a_repeat": (
        "off-by-one repeat: each A row is sent one time fewer than the cores consume it",
        ".repeat(N // n // n_aie_cols)",
        ".repeat(N // n // n_aie_cols - 1)",
    ),
    "split_offset": (
        "wrong split offsets: memtile rows rotated",
        "[m * k * j for j in range(stop_row - start_row)]",
        "[m * k * ((j + 1) % (stop_row - start_row)) for j in range(stop_row - start_row)]",
    ),
    "a_row_wiring": (
        "A split outputs wired to the wrong core row: each row computes with its neighbour's A",
        "A_l2l1_fifos[row].cons(),",
        "A_l2l1_fifos[(row + 1) % n_aie_rows].cons(),",
    ),
    "c_double_write": (
        "C tap ignores the row block: every block drains over block 0",
        'transfers.append(("C", col, tilings.C[row, col::n_aie_cols]))',
        'transfers.append(("C", col, tilings.C[row * 0, col::n_aie_cols]))',
    ),
    "b_permute": (
        "B tiles walked row of tiles first instead of column of tiles first",
        "TensorAccessPattern.full((K, N)).tile((k, n)).permute((1, 0, 2, 3))",
        "TensorAccessPattern.full((K, N)).tile((k, n))",
    ),
    "a_layout": (
        "A streamed to the cores row-major instead of in (r, s) micro-tiles",
        "to_stream=[dims.A] * (stop_row - start_row)",
        "to_stream=[None] * (stop_row - start_row)",
    ),
}


class MutantLake:
    """A scratch copy of the Lake project sharing the in-tree packages."""

    def __init__(self, root: Path):
        self.root = root
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for name in ("lakefile.toml", "lean-toolchain", "lake-manifest.json"):
            shutil.copy(LEAN_ROOT / name, root / name)
        for name in ("IronEval.lean", "IronVerify.lean"):
            shutil.copy(LEAN_ROOT / name, root / name)
        shutil.copytree(LEAN_ROOT / "IronVerify", root / "IronVerify")
        (root / ".lake").mkdir()
        (root / ".lake/packages").symlink_to(LEAN_ROOT / ".lake/packages")

    def lake(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["lake", *args], cwd=self.root, capture_output=True, text=True
        )

    def proof_check(self, design: str) -> str:
        """Build the WholeArrayC1 proofs retargeted at `design`: `builds`, `fails` or why not run."""
        proofs = self.root / "IronVerify/Proofs"
        original = proofs / "WholeArrayC1.lean"
        if not original.exists():
            return "not run: Proofs/WholeArrayC1.lean does not exist yet"
        target = proofs / f"{design}.lean"
        if design != "WholeArrayC1":
            target.write_text(original.read_text().replace("WholeArrayC1", design))
        result = self.lake("build", f"IronVerify.Proofs.{design}")
        log = result.stdout + result.stderr
        if result.returncode != 0:
            first = next((l for l in log.splitlines() if "error" in l), "")
            return f"fails ({first.strip()[:120]})"
        if "declaration uses `sorry`" in log or "declaration uses 'sorry'" in log:
            return "builds, but the proofs still use sorry (inconclusive)"
        return "builds"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="scratch Lake copy (default: a temp dir)",
    )
    p.add_argument("--only", nargs="*", choices=sorted(MUTATIONS), default=None)
    p.add_argument(
        "--shapes", nargs="+", default=["256x128x64", "512x128x64"], metavar="MxKxN"
    )
    p.add_argument("--skip-proofs", action="store_true")
    p.add_argument("--device", choices=["npu1", "npu2"], default="npu2")
    opts = p.parse_args()

    m, k, n, cols = 64, 64, 32, 1
    iron.set_current_device(from_name(opts.device, n_cols=None))
    mac_dims = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n, input_dtype=np.int16, output_dtype=np.int16
    ).mac_dims

    work = opts.work_dir or Path(tempfile.mkdtemp(prefix="iron_lean_mutants_"))
    lake = MutantLake(work / "lake")
    source_text = WHOLE_ARRAY.read_text()
    cases = {"WholeArrayC1": ("unmutated whole_array", None, None)}
    for name in opts.only if opts.only is not None else sorted(MUTATIONS):
        title = "Mut" + "".join(part.capitalize() for part in name.split("_"))
        cases[title] = MUTATIONS[name]

    sources = {}
    rows = []
    for design, (title, old, new) in cases.items():
        source = WHOLE_ARRAY
        if old is not None:
            assert source_text.count(old) == 1, f"{design}: mutation site not unique"
            source = work / f"whole_array_{design}.py"
            source.write_text(source_text.replace(old, new))
        try:
            program = whole_array_program(source, None, m, k, n, cols, opts.device)
        except ValueError as exc:
            print(
                f"== {design}: {title}\n  rejected by IRON at resolve: {str(exc)[:160]}"
            )
            rows.append((design, "-", "-", f"rejected at resolve: {str(exc)[:60]}"))
            continue
        sources[design] = source
        write_design(program, design, lake.root)
    build = lake.lake("build", "ironeval")
    assert build.returncode == 0, build.stdout + build.stderr

    for design, source in sources.items():
        title = cases[design][0]
        print(f"== {design}: {title}")
        model = compare(
            source, lake.root, design, opts.shapes, m, k, n, cols, opts.device
        )
        problems = []
        for shape in opts.shapes:
            M, K, N = (int(x) for x in shape.split("x"))
            program = whole_array_program(source, shape, m, k, n, cols, opts.device)
            for error in MatmulCheck(program, M, K, N, m, k, n, mac_dims).run():
                problems.append(f"{shape}: {error}")
        for line in problems[:4]:
            print(f"  matmul: {line}")
        proof = "skipped" if opts.skip_proofs else lake.proof_check(design)
        rows.append(
            (design, "PASS" if model else "FAIL", "FAIL" if problems else "PASS", proof)
        )

    print(f"\n{'design':<20} {'difftest':<9} {'A @ B':<6} proofs")
    for design, model, matmul, proof in rows:
        print(f"{design:<20} {model:<9} {matmul:<6} {proof}")
    expected = all(
        proof.startswith("rejected")
        or (
            model == "PASS"
            and (matmul == "PASS") == (design == "WholeArrayC1")
            and (
                proof.startswith("builds") == (design == "WholeArrayC1")
                or proof.startswith(("not run", "skipped"))
            )
        )
        for design, model, matmul, proof in rows
    )
    print("every mutant caught, original verified" if expected else "UNEXPECTED RESULT")
    raise SystemExit(0 if expected else 1)


if __name__ == "__main__":
    main()
