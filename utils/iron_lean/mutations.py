# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Mutation suite: inject real bugs into whole_array and check each layer of the verifier sees them.

Each mutant is a copy of whole_array.py with one textual change, exported in one configuration.
For every mutant, and for the unmutated design in each configuration a mutant uses, this runs:

* the design difftest (Lean evaluation of the symbolic export vs the numpy oracle). It checks
  the *model*, so a mutant is expected to PASS here too: both sides describe the same, buggy,
  design;
* the concrete matmul check (oracle data movement + each core's kernel contract vs ``A @ B``),
  which must FAIL for a mutant;
* the instance check: the export, generated exactly as instances.py generates a shipped
  configuration, must prove to be an ``Instance`` of the parametric schema. It must fail for a
  mutant; the report names the parts that fail (fifo graph, kernel layouts, a channel's stream,
  the guards).

Mutant Lean files live in a scratch copy of the Lake project (`--work-dir`) whose
`.lake/packages` links to the in-tree one, so the in-tree project is untouched.
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from difftest_design import compare
from export import LEAN_ROOT, WHOLE_ARRAY, whole_array_program, write_design
from instances import (
    INSTANCES,
    Config,
    Instance,
    check_module,
    export_instance,
    streams_module,
)
from matmul_check import MatmulCheck

DEFAULT = Config("npu2", "i16", "i16", 64, 64, 32, 0, 0, 0, "whole_array.py defaults")
B_COL_MAJ = Config(
    "npu2", "i16", "i32", 32, 128, 32, 1, 0, 0, "run_strix_makefile_col_maj"
)


@dataclass(frozen=True)
class Mutation:
    """One textual bug and the configuration that exercises it."""

    title: str
    old: str
    new: str
    config: Config = DEFAULT
    cols: int = 1
    shapes: tuple[str, ...] = ("256x128x64", "512x128x64")


MUTATIONS = {
    "a_permute": Mutation(
        "wrong A permute: the repeat dim stays outermost, so A[row] selects a repeat",
        ".repeat(N // n // n_aie_cols)\n        .permute((1, 0, 2, 3, 4))",
        ".repeat(N // n // n_aie_cols)\n        .permute((0, 1, 2, 3, 4))",
    ),
    "a_repeat": Mutation(
        "off-by-one repeat: each A row is sent one time fewer than the cores consume it",
        ".repeat(N // n // n_aie_cols)",
        ".repeat(N // n // n_aie_cols - 1)",
    ),
    "split_offset": Mutation(
        "wrong split offsets: memtile rows rotated",
        "[m * k * j for j in range(rows)]",
        "[m * k * ((j + 1) % rows) for j in range(rows)]",
    ),
    "a_row_wiring": Mutation(
        "A split outputs wired to the wrong core row: each row computes with its neighbour's A",
        "A_l2l1_fifos[row].cons(),",
        "A_l2l1_fifos[(row + 1) % n_aie_rows].cons(),",
    ),
    "c_double_write": Mutation(
        "C tap ignores the row block: every block drains over block 0",
        'transfers.append(("C", col, tilings.C[row, col::n_aie_cols]))',
        'transfers.append(("C", col, tilings.C[row * 0, col::n_aie_cols]))',
    ),
    "b_permute": Mutation(
        "B tiles walked row of tiles first instead of column of tiles first",
        "TensorAccessPattern.full((K, N)).tile((k, n)).permute((1, 0, 2, 3))",
        "TensorAccessPattern.full((K, N)).tile((k, n))",
    ),
    "a_layout": Mutation(
        "A streamed to the cores row-major instead of in (r, s) micro-tiles",
        "to_stream=[dims.A] * rows",
        "to_stream=[None] * rows",
    ),
    "a_uneven_tap": Mutation(
        "A tap assumes the core rows split evenly over the A shims (wrong only at 3 columns)",
        "A_blocks[:, :, first * m : (first + rows) * m]",
        "A_blocks[:, :, first * m : (first + n_aie_rows // min(n_aie_rows, n_aie_cols)) * m]",
        cols=3,
        shapes=("256x128x96", "512x128x96"),
    ),
    "b_col_maj_tap": Mutation(
        "column-major B walked k tile first (wrong only with b_col_maj)",
        "B_tiles = TensorAccessPattern.full((N, K)).tile((n, k))\n",
        "B_tiles = TensorAccessPattern.full((N, K)).tile((n, k)).permute((1, 0, 2, 3))\n",
        config=B_COL_MAJ,
        shapes=("256x256x64", "512x256x64"),
    ),
}

# The parts of an `Instance` proof, by the theorem instances.py generates for each.
PARTS = {
    "net_eq": "fifo graph",
    "kernels_ok": "kernel layouts",
    "guard": "guards",
    "chans_eq": "channels",
    "every": "channels",
    "inst": "walk sizes",
}


class MutantLake:
    """A scratch copy of the Lake project sharing the in-tree packages and build."""

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
        # A copy, not links: rebuilding here must never rewrite the in-tree artifacts.
        shutil.copytree(LEAN_ROOT / ".lake/build", root / ".lake/build", symlinks=True)

    def lake(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["lake", *args], cwd=self.root, capture_output=True, text=True
        )

    def failing(self, module: str, text: str) -> list[str] | None:
        """Build `module` (whose source is `text`): None if it builds, else its failing theorems."""
        result = self.lake("build", module)
        if result.returncode == 0:
            return None
        path = module.replace(".", "/") + ".lean"
        lines = text.splitlines()
        names = []
        for match in re.finditer(
            rf"error: {re.escape(path)}:(\d+):\d+:", result.stdout + result.stderr
        ):
            before = lines[: int(match.group(1))]
            name = next(
                (
                    m.group(1)
                    for l in reversed(before)
                    if (m := re.match(r"theorem (\S+)", l))
                ),
                "(module header)",
            )
            if name not in names:
                names.append(name)
        return names or ["(build failed outside the module)"]

    def instance_check(self, inst: Instance) -> str:
        """Prove `inst` is an `Instance` of the schema: `instance`, or the parts that fail."""
        gen = self.root / INSTANCES
        gen.mkdir(parents=True, exist_ok=True)
        module = "IronVerify.Schema.Instances." + inst.name
        streams = streams_module(inst)
        check = check_module(inst, inst)
        (gen / f"{inst.name}.lean").write_text(inst.export)
        (gen / f"{inst.name}Streams.lean").write_text(streams)
        failed_streams = self.failing(f"{module}Streams", streams) or []
        if failed_streams:
            # Without the channel proofs, check the rest on its own.
            check = re.sub(
                r"(?ms)^theorem chan_streams.*?(?=^theorem guard)", "", check
            )
            check = re.sub(r"(?ms)^theorem inst\b.*?(?=^end )", "", check)
            check = check.replace(
                f"import IronVerify.Schema.Instances.{inst.name}Streams\n",
                "import IronVerify.Schema.Inst.Tactic\n",
            )
        (gen / f"{inst.name}Check.lean").write_text(check)
        failed = self.failing(f"{module}Check", check) or []
        parts = [
            "stream " + name.removeprefix("ch_")
            for name in failed_streams
            if name.startswith("ch_")
        ] + [n for n in failed_streams if not n.startswith("ch_")]
        parts += [PARTS.get(name, name) for name in failed]
        if not parts:
            return "instance"
        return "fails: " + ", ".join(dict.fromkeys(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="scratch Lake copy (default: a temp dir)",
    )
    p.add_argument("--only", nargs="*", choices=sorted(MUTATIONS), default=None)
    p.add_argument("--skip-instances", action="store_true")
    opts = p.parse_args()

    work = opts.work_dir or Path(tempfile.mkdtemp(prefix="iron_lean_mutants_"))
    lake = MutantLake(work / "lake")
    source_text = WHOLE_ARRAY.read_text()
    selected = [
        (name, MUTATIONS[name])
        for name in (opts.only if opts.only is not None else sorted(MUTATIONS))
    ]
    cases: list[tuple[str, str, Path, Mutation]] = []
    for config, cols, shapes in dict.fromkeys(
        (m.config, m.cols, m.shapes) for _, m in selected
    ):
        cases.append(
            (
                "Orig_" + config.name(cols),
                "unmutated whole_array",
                WHOLE_ARRAY,
                Mutation("unmutated", "", "", config, cols, shapes),
            )
        )
    for name, mutation in selected:
        assert source_text.count(mutation.old) == 1, f"{name}: mutation site not unique"
        source = work / f"whole_array_{name}.py"
        source.write_text(source_text.replace(mutation.old, mutation.new))
        design = "Mut" + "".join(part.capitalize() for part in name.split("_"))
        cases.append((design, mutation.title, source, mutation))

    rows = []
    exported = []
    for design, title, source, mutation in cases:
        c = mutation.config
        options = dict(
            dtype_in=c.dtype_in,
            dtype_out=c.dtype_out,
            b_col_maj=c.b_col_maj,
            c_col_maj=c.c_col_maj,
            emulate_bf16=c.emulate,
        )
        try:
            program = whole_array_program(
                source, None, c.m, c.k, c.n, mutation.cols, c.device, **options
            )
        except ValueError as exc:
            print(
                f"== {design}: {title}\n  rejected by IRON at resolve: {str(exc)[:160]}"
            )
            rows.append((design, "-", "-", f"rejected at resolve: {str(exc)[:60]}"))
            continue
        write_design(program, design, lake.root)
        exported.append((design, title, source, mutation, options))
    build = lake.lake("build", "ironeval")
    assert build.returncode == 0, build.stdout + build.stderr

    for design, title, source, mutation, options in exported:
        c = mutation.config
        print(f"== {design}: {title}")
        model = compare(
            source,
            lake.root,
            design,
            list(mutation.shapes),
            c.m,
            c.k,
            c.n,
            mutation.cols,
            c.device,
            **options,
        )
        problems = []
        for shape in mutation.shapes:
            M, K, N = (int(x) for x in shape.split("x"))
            program = whole_array_program(
                source, shape, c.m, c.k, c.n, mutation.cols, c.device, **options
            )
            check = MatmulCheck(program, M, K, N, bool(c.b_col_maj), bool(c.c_col_maj))
            problems += [f"{shape}: {error}" for error in check.run()]
        for line in problems[:4]:
            print(f"  matmul: {line}")
        instance = "skipped"
        if not opts.skip_instances:
            inst = export_instance(c, mutation.cols, source, design)
            instance = lake.instance_check(inst)
            print(f"  instance: {instance}")
        rows.append(
            (
                design,
                "PASS" if model else "FAIL",
                "FAIL" if problems else "PASS",
                instance,
            )
        )

    print(f"\n{'design':<36} {'difftest':<9} {'A @ B':<6} schema instance")
    for design, model, matmul, instance in rows:
        print(f"{design:<36} {model:<9} {matmul:<6} {instance}")
    expected = all(
        instance.startswith("rejected")
        or (
            model == "PASS"
            and (matmul == "PASS") == design.startswith("Orig_")
            and (
                (instance == "instance") == design.startswith("Orig_")
                or instance == "skipped"
            )
        )
        for design, model, matmul, instance in rows
    )
    print("every mutant caught, original verified" if expected else "UNEXPECTED RESULT")
    raise SystemExit(0 if expected else 1)


if __name__ == "__main__":
    main()
