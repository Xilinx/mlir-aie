# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import subprocess
import sys

from provenance_checks import provenance


def generate(source, expected):
    from aie import ir
    from aie.iron import Program, Runtime, Worker, WorkerRuntimeBarrier
    from aie.iron.device import NPU2Col1

    barrier = WorkerRuntimeBarrier()

    def task(barrier):
        barrier.wait_for_value(1)
        barrier.release_with_value(1)

    def sequence():
        barrier.set(1)

    module = Program(
        NPU2Col1(), Runtime(sequence, []), workers=[Worker(task, fn_args=[barrier])]
    ).resolve_program()
    attr = ir.DictAttr(module.operation.attributes["iron.provenance"])
    values = {
        name: attr[name].value if name in attr else None
        for name in ("version", "revision")
    }
    # Save the producer's metadata before the separate compiler process runs.
    pathlib.Path(source).write_text(str(module))
    pathlib.Path(expected).write_text(json.dumps(values))


def check(directory, expected):
    root = pathlib.Path(directory)
    wanted = json.loads(pathlib.Path(expected).read_text())
    for name in ("full.elf", "partial.elf"):
        info = provenance((root / name).read_bytes())
        assert info["iron"] == wanted, info
    metadata = json.loads((root / "metadata.json").read_text())
    entries = {
        e["key"]: e["value"] for e in metadata["keyvalue_metadata"]["key_values"]
    }
    assert json.loads(entries["mlir_aie.provenance"])["iron"] == wanted


def validate(directory):
    root = pathlib.Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        ("", None),
        ('iron.entry = "main:sequence"', None),
        ("iron.provenance = {}", dict(version=None, revision=None)),
        (
            'iron.provenance = {version = "old-bindings", revision = "abcdef0"}',
            dict(version="old-bindings", revision="abcdef0"),
        ),
        ('iron.provenance = "bad"', "must be a dictionary"),
        ("iron.provenance = {version = 1 : i32}", "invalid iron.provenance field"),
        ('iron.provenance = {unknown = "value"}', "invalid iron.provenance field"),
    ]
    for i, (attrs, expected) in enumerate(cases):
        source = root / f"case{i}.mlir"
        source.write_text(f"module attributes {{{attrs}}} {{}}")
        output = root / str(i)
        result = subprocess.run(
            [
                "aiecc",
                "--get=provenance.json",
                f"--output-dir={output}",
                f"--tmpdir={output}/work",
                str(source),
            ],
            text=True,
            capture_output=True,
        )
        if isinstance(expected, str):
            assert result.returncode != 0 and expected in result.stderr, result
        else:
            assert result.returncode == 0, result.stderr
            info = json.loads((output / "provenance.json").read_text())
            if expected is None:
                assert "iron" not in info
            else:
                assert info["iron"] == expected


if __name__ == "__main__":
    {"generate": generate, "check": check, "validate": validate}[sys.argv[1]](
        *sys.argv[2:]
    )
