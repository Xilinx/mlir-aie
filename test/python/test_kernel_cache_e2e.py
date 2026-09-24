# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""A cached kernel object links into the same binaries a cold build produces.

Each build runs the JIT in a fresh process against its own ``NPU_CACHE_HOME``,
so only the on-disk caches carry over between builds. Both flows are covered:
an xclbin with its instruction stream, and a full ELF holding the PDI and the
control code. The xclbin is left out of every comparison: xclbinutil stamps
each one it writes, even with no cache at all.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Each flow's packaging tool, and the artifacts it adds to the ones both share.
_FLOWS = {
    "xclbin": ("xclbinutil", ["insts.bin"]),
    "full_elf": ("aiebu-asm", ["npu_insts_full_elf_*.bin", "design.elf"]),
}

_DESIGN = textwrap.dedent("""
    import json, logging, sys
    from collections import Counter
    from pathlib import Path
    import numpy as np
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2Col1
    from aie.utils import set_current_device

    source = Path(sys.argv[1])
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    @iron.jit
    def add_one(input: In, output: Out, *, n: CompileTime[int]):
        kernel = ExternalFunction(
            "add_one",
            source_file=str(source),
            include_dirs=[str(source.parent)],
            arg_types=[tile_ty, tile_ty, np.int32],
        )
        tensor_ty = np.ndarray[(n,), np.dtype[np.int32]]
        of_in, of_out = ObjectFifo(tile_ty, name="in"), ObjectFifo(tile_ty, name="out")

        def core_body(i, o, k):
            for _ in range_(n // 16):
                a, b = i.acquire(1), o.acquire(1)
                k(a, b, 16)
                i.release(1)
                o.release(1)

        worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), kernel])

        def seq(a, b, i, o):
            i.fill(a)
            o.drain(b, wait=True)

        rt = Runtime(seq, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    lookups = Counter()

    class Count(logging.Handler):
        def __init__(self, prefix):
            super().__init__()
            self.prefix = prefix

        def emit(self, record):
            lookups[self.prefix + record.getMessage().split(" for ")[0].split()[-1]] += 1

    for module, prefix in (("_object_cache", ""), ("_explicit_builds", "build_")):
        log = logging.getLogger(f"aie.utils.compile.jit.{module}")
        log.setLevel(logging.DEBUG)
        log.addHandler(Count(prefix))
    set_current_device(NPU2Col1())
    design = add_one.specialize(
        n=int(sys.argv[2]),
        use_cache=sys.argv[3] == "warm",
        full_elf=sys.argv[4] == "full_elf",
    )
    out = Path(sys.argv[5]) if len(sys.argv) > 5 else None
    if out is None:
        design.compile()
    elif design.compilable.full_elf:
        design.compile(full_elf_path=out / "design.elf")
    else:
        design.compile(xclbin_path=out / "final.xclbin", inst_path=out / "insts.bin")
    compilable = design.compilable
    print(json.dumps({
        "dir": str(compilable._kernel_dir),
        "kernel": compilable._full_elf_kernel_name,
        "sizes": compilable._expected_tensor_sizes,
        **lookups,
    }))
    """)

_SOURCE = 'extern "C" void add_one(int *i, int *o, int n) { for (int j = 0; j < n; j++) o[j] = i[j] + STEP; }\n'


@pytest.fixture(
    params=[
        pytest.param(
            flow, marks=pytest.mark.skipif(shutil.which(tool) is None, reason=tool)
        )
        for flow, (tool, _) in _FLOWS.items()
    ]
)
def flow(request):
    return request.param


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "add_one.cc").write_text('#include "step.h"\n' + _SOURCE)
    _set_step(src / "add_one.cc", 1)
    return src / "add_one.cc"


def _set_step(source, step):
    (source.parent / "step.h").write_text(f"#define STEP {step}\n")


def _run(tmp_path, source, flow, home, n=32, mode="warm", out=None):
    script = tmp_path / "design.py"
    script.write_text(_DESIGN)
    named = [str(out)] if out is not None else []
    result = subprocess.run(
        [sys.executable, str(script), str(source), str(n), mode, flow, *named],
        env={**os.environ, "NPU_CACHE_HOME": str(tmp_path / home)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def _build(tmp_path, source, flow, home, n=32, mode="warm", out=None):
    build = _run(tmp_path, source, flow, home, n, mode, out)
    return Path(build["dir"]), build.get("hit", 0), build.get("miss", 0)


def _binaries(kernel_dir, flow):
    """Every deterministic artifact from the kernel object to the device image."""
    names = [
        "add_one.o",
        "elfs_*/*.elf",
        "cdo_main/*.bin",
        "main.pdi",
        *_FLOWS[flow][1],
    ]
    found = {
        str(path.relative_to(kernel_dir)): path.read_bytes()
        for pattern in names
        for path in kernel_dir.glob(pattern)
    }
    assert len(found) >= len(names)
    return found


def _objects(tmp_path, home):
    return list((tmp_path / home / "objects").iterdir())


def test_a_cached_object_links_into_the_binaries_a_cold_build_makes(
    tmp_path, source, flow
):
    first, hits, misses = _build(tmp_path, source, flow, "warm", n=32)
    assert (hits, misses) == (0, 1)
    second, hits, misses = _build(tmp_path, source, flow, "warm", n=64)
    assert (hits, misses) == (1, 0)
    assert first != second
    assert len(_objects(tmp_path, "warm")) == 1

    cold, hits, misses = _build(tmp_path, source, flow, "cold", n=64, mode="cold")
    assert (hits, misses) == (0, 0)
    assert not (tmp_path / "cold" / "objects").exists()
    assert _binaries(second, flow) == _binaries(cold, flow)


def test_a_header_edit_reaches_the_core_elf(tmp_path, source, flow):
    """A header is not part of the key: the entry notices the edit and rebuilds."""
    kernel_dir, _, _ = _build(tmp_path, source, flow, "warm")
    step_one = _binaries(kernel_dir, flow)
    entries = _objects(tmp_path, "warm")

    _set_step(source, 2)
    edited, hits, misses = _build(tmp_path, source, flow, "warm")
    assert (hits, misses) == (0, 1)
    assert (edited, _objects(tmp_path, "warm")) == (kernel_dir, entries)
    step_two = _binaries(edited, flow)
    changed = {name for name in step_one if step_one[name] != step_two[name]}
    assert changed >= {
        "add_one.o",
        "elfs_main_core_0_2/elfs_main_core_0_2.elf",
        "cdo_main/main_aie_cdo_elfs.bin",
        "main.pdi",
    }
    if flow == "full_elf":
        assert "design.elf" in changed
    cold, _, _ = _build(tmp_path, source, flow, "cold", mode="cold")
    assert _binaries(cold, flow) == step_two

    # Step one's device is still in aiecc's device cache, so its core is not
    # compiled again and leaves no core ELF behind.
    _set_step(source, 1)
    reverted, hits, misses = _build(tmp_path, source, flow, "warm")
    assert (hits, misses) == (0, 1)
    assert (
        len([p for p in (tmp_path / "warm" / "devices").iterdir() if p.is_dir()]) == 2
    )
    assert _binaries(reverted, flow) == {
        name: data for name, data in step_one.items() if not name.startswith("elfs_")
    }


def _named_outputs(out, prj):
    """The outputs a caller named, and the PDI; an xclbin differs every build."""
    named = {
        p.name: p.read_bytes() for p in out.iterdir() if p.suffix in (".bin", ".elf")
    }
    return {**named, "main.pdi": (prj / "main.pdi").read_bytes()}


def test_named_outputs_are_rebuilt_exactly_when_out_of_date(tmp_path, source, flow):
    """Named outputs are checked in place before any cache is consulted.

    A build that reuses its outputs looks up no kernel object. A header is in
    no key, so only the recorded inputs can catch its edit.
    """
    out = tmp_path / "out"
    prj, hits, misses = _build(tmp_path, source, flow, "warm", out=out)
    assert (hits, misses) == (0, 1)
    step_one = _named_outputs(out, prj)
    assert _build(tmp_path, source, flow, "warm", out=out) == (prj, 0, 0)

    _set_step(source, 2)
    assert _build(tmp_path, source, flow, "warm", out=out) == (prj, 0, 1)
    step_two = _named_outputs(out, prj)
    assert step_two != step_one
    cold, _, _ = _build(tmp_path, source, flow, "cold", mode="cold")
    assert step_two == {name: _binaries(cold, flow)[name] for name in step_two}
    assert _build(tmp_path, source, flow, "warm", out=out) == (prj, 0, 0)

    # Anything else rewriting an output makes it unknown, even to the same
    # bytes, so the build is fetched again.
    output = next(p for p in out.iterdir() if p.suffix in (".bin", ".elf"))
    for data in (output.read_bytes(), b""):
        output.write_bytes(data)
        rebuilt = _run(tmp_path, source, flow, "warm", out=out)
        assert (rebuilt["dir"], rebuilt.get("build_hit")) == (str(prj), 1)
        assert "hit" not in rebuilt and "miss" not in rebuilt
        assert _named_outputs(out, prj) == step_two

    assert _build(tmp_path, source, flow, "warm", n=64, out=out) == (prj, 1, 0)
    assert _named_outputs(out, prj) != step_two

    _set_step(source, 1)
    assert _build(tmp_path, source, flow, "warm", n=32, out=out) == (prj, 0, 1)
    assert _named_outputs(out, prj) == step_one


def _work_files(prj):
    names = ("input_with_addresses.mlir", "full_elf_config.json", "main.pdi")
    return {name: (prj / name).read_bytes() for name in names if (prj / name).is_file()}


def test_a_named_build_is_fetched_into_a_new_output_dir(tmp_path, source, flow):
    """The same design built to a new path copies the build instead of rerunning it.

    Every output is fetched, the xclbin included, along with the work files a
    caller reads afterwards. A header edit is a miss, like everywhere else, and
    so is a build that does not use the cache.
    """
    first = _run(tmp_path, source, flow, "warm", out=tmp_path / "a")
    assert (first.get("miss"), first.get("build_hit")) == (1, None)
    second = _run(tmp_path, source, flow, "warm", out=tmp_path / "b")
    assert second.keys() == {"dir", "kernel", "sizes", "build_hit"}
    assert (second["kernel"], second["sizes"]) == (first["kernel"], first["sizes"])
    if flow == "full_elf":
        assert second["kernel"]
    a, b = Path(first["dir"]), Path(second["dir"])
    assert a != b
    assert _work_files(b) == _work_files(a) != {}
    outputs = {
        p.name: p.read_bytes() for p in (tmp_path / "a").iterdir() if p.is_file()
    }
    assert {
        p.name: p.read_bytes() for p in (tmp_path / "b").iterdir() if p.is_file()
    } == outputs
    assert _build(tmp_path, source, flow, "warm", out=tmp_path / "b") == (b, 0, 0)

    cold = _run(tmp_path, source, flow, "cold", mode="cold", out=tmp_path / "c")
    assert "build_hit" not in cold
    assert not (tmp_path / "cold" / "builds").exists()

    _set_step(source, 2)
    edited = _run(tmp_path, source, flow, "warm", out=tmp_path / "d")
    assert (edited.get("miss"), edited.get("build_hit")) == (1, None)
    step_two = _named_outputs(tmp_path / "d", Path(edited["dir"]))
    assert step_two != _named_outputs(tmp_path / "a", a)
    fetched = _run(tmp_path, source, flow, "warm", out=tmp_path / "e")
    assert fetched.get("build_hit") == 1
    assert _named_outputs(tmp_path / "e", Path(fetched["dir"])) == step_two
