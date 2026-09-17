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
kernels reach theirs through a generated ``#include``, and every file's own
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
# Fused GEMM exposes a multi-entry-point ABI driven by an external design's
# loop nest; it arrived on main without a Python factory.
_DESIGN_ONLY = {_KERNELS / "generic" / "mm_fused.cc"}


def _built_by_a_factory() -> set[Path]:
    """Return the sources some factory compiles, over both architectures."""
    built: set[Path] = set()
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
                # The aie2 LUT kernels are compiled from a generated source
                # that includes the .cc rather than naming it as source_file.
                for line in (getattr(fn, "source_string", None) or "").splitlines():
                    if '#include "' in line:
                        path = Path(line.split('"')[1])
                        if path.suffix == ".cc":
                            built.add(path.resolve())
    return built


def _included_by_another_kernel(source: Path) -> bool:
    """Whether some kernel ``#include``s this one, by bare name or by path."""
    hits = subprocess.run(
        ["grep", "-rlE", rf'#include "(.*/)?{source.name}"', str(_KERNELS)],
        capture_output=True,
        text=True,
    ).stdout.split()
    return any(Path(h).resolve() != source.resolve() for h in hits)


@pytest.mark.skipif(not _KERNELS.is_dir(), reason="no aie_kernels/ checkout")
def test_no_kernel_source_is_unreachable():
    built = _built_by_a_factory()
    sources = sorted(_KERNELS.rglob("*.cc"))
    # A factory resolves its source against MLIR_AIE_KERNEL_SOURCES, which may
    # point at an installed copy rather than this checkout. Then no path here
    # matches and every kernel looks orphaned, which is a misconfigured probe
    # and not a tree full of dead code -- so say which it is.
    assert built & {p.resolve() for p in sources}, (
        f"no factory resolved to a source under {_KERNELS}, so this cannot tell "
        "a reachable kernel from an orphaned one. Point MLIR_AIE_KERNEL_SOURCES "
        "at this checkout and rerun."
    )
    unreachable = sorted(
        p
        for p in sources
        if p not in _DESIGN_ONLY
        and p.resolve() not in built
        and not _included_by_another_kernel(p)
    )
    assert not unreachable, (
        "no factory compiles these, so nothing ever checks them: "
        f"{[str(p.relative_to(_KERNELS)) for p in unreachable]}. Give each one a "
        "factory, or delete it and its row in aie_kernels/README.md."
    )
