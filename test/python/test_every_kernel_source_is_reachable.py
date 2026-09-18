# test_every_kernel_source_is_reachable.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Every file in ``aie_kernels/`` is compiled by some factory.

A kernel source nothing builds is a copy of something, drifting: it is never
compiled, so it never fails, and the catalogue in aie_kernels/README.md goes
on advertising it. That is how bf16_softmax.cc outlived the deduplication
(#3599) that removed its last caller while softmax.cc kept providing the same
bf16 softmax.

Asking the factories is the only reliable way to tell. A grep cannot: a
factory names its source by computed string (``f"bitwise{op}.cc"``), the LUT
kernels reach theirs through a source-selection define, and every file's own
LLVM banner mentions its filename.
"""

import inspect
import subprocess
from pathlib import Path

import aie.iron as iron
import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1

_KERNELS = Path(__file__).resolve().parents[2] / "aie_kernels"


def _built_by_a_factory() -> set[Path]:
    """Return the sources some factory compiles, over both architectures."""
    built: set[Path] = set()
    previous = iron.get_current_device(probe_runtime=False)
    try:
        for device in (NPU1Col1, NPU2Col1):
            iron.set_current_device(device())
            for name in kernels.__all__:
                factory = getattr(kernels, name)
                if not inspect.isfunction(factory) or name.endswith("_ref"):
                    continue
                for combo in [{}] + [dict(c) for c in getattr(factory, "dtypes", ())]:
                    try:
                        fn = factory(**combo)
                    except Exception:  # noqa: BLE001 - needs kwargs, or wrong arch
                        continue
                    if getattr(fn, "source_file", None):
                        built.add(Path(fn.source_file).resolve())
                    # The AIE2 LUT translation unit selects its native kernel
                    # source via a preprocessor include operand.
                    for flag in fn.compile_flags:
                        if flag.startswith("-DAIE_LUT_KERNEL_SOURCE="):
                            built.add(Path(flag.split("=", 1)[1].strip('"')).resolve())
    finally:
        iron.set_current_device(previous)
    return built


@pytest.mark.parametrize("device", [None, NPU1Col1, NPU2Col1])
@pytest.mark.parametrize("fails", [False, True])
def test_factory_probe_restores_device(monkeypatch, device, fails):
    previous = iron.get_current_device(probe_runtime=False)
    selected = device() if device else None
    iron.set_current_device(selected)
    try:
        monkeypatch.setattr(kernels, "__all__", ["missing_factory"] if fails else [])
        if fails:
            with pytest.raises(AttributeError, match="missing_factory"):
                _built_by_a_factory()
        else:
            assert _built_by_a_factory() == set()
        assert iron.get_current_device(probe_runtime=False) is selected
    finally:
        iron.set_current_device(previous)


def _included_by_another_kernel(source: Path) -> bool:
    """Whether some kernel ``#include``s this one, by bare name or by path."""
    hits = subprocess.run(
        ["grep", "-rlE", rf'#include "(.*/)?{source.name}"', str(_KERNELS)],
        capture_output=True,
        text=True,
    ).stdout.split()
    return any(Path(h).resolve() != source.resolve() for h in hits)


@pytest.mark.skipif(not _KERNELS.is_dir(), reason="no aie_kernels/ checkout")
def test_no_kernel_source_is_unreachable(monkeypatch):
    # Probe this checkout even when lit imports the package from a staged build.
    monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(_KERNELS.parent))
    built = _built_by_a_factory()
    sources = sorted(_KERNELS.rglob("*.cc"))
    assert built & {p.resolve() for p in sources}, (
        f"no factory resolved to a source under {_KERNELS}, so this cannot tell "
        "a reachable kernel from an orphaned one. Point MLIR_AIE_KERNEL_SOURCES "
        "at this checkout and rerun."
    )
    unreachable = sorted(
        p
        for p in sources
        if p.resolve() not in built and not _included_by_another_kernel(p)
    )
    assert not unreachable, (
        "no factory compiles these, so nothing ever checks them: "
        f"{[str(p.relative_to(_KERNELS)) for p in unreachable]}. Give each one a "
        "factory, or delete it and its row in aie_kernels/README.md."
    )
