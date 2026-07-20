# vector_scalar_add/vector_scalar_add.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vector scalar add — IRON API design with ``@iron.jit`` compilation.

A single AIE compute core adds 1 to each element of an ``int32`` vector.
The body delegates to ``aie.iron.algorithms.transform`` with an
inline lambda — the algorithm handles the ObjectFifo / Worker / Runtime
plumbing including the memtile staging tile.

Invocation modes:

  * standalone:      ``python3 vector_scalar_add.py``
  * compile-only:    ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
  * artifact export: ``... --aot-dir=DIR``  (writes named xclbin / insts /
    PDI / ELF into DIR and prints where each landed — see ``aot_compile``)
  * bring-your-own:  ``... --from-xclbin=PATH --from-insts=PATH``  (loads a
    pre-built xclbin + insts — from any toolchain — and runs it; see
    ``run_from_artifacts``.  Pass ``--dev`` to assert the attached NPU family
    matches the artifacts, since a mismatched xclbin typically hangs or times out.)
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform
from aie.utils import NPUKernel
from aie.utils.compile import resolve_target_arch
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def vector_scalar_add(
    inp: In,
    out: Out,
    *,
    problem_size: CompileTime[int] = 1024,
    aie_tile_width: CompileTime[int] = 32,
):
    tensor_ty = np.ndarray[(problem_size,), np.dtype[np.int32]]
    return transform(lambda x: x + 1, tensor_ty, tile_size=aie_tile_width)


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Scalar Add")
    add_compile_args(p, with_elf=True, with_pdi=True)
    p.add_argument(
        "--problem-size",
        type=int,
        default=1024,
        help="total elements in the input vector",
    )
    p.add_argument(
        "--aie-tile-width",
        type=int,
        default=32,
        help="elements per compute-tile sub-tile (transform tile_size)",
    )
    p.add_argument(
        "--aot-dir",
        type=str,
        default=None,
        help="ahead-of-time compile: write named xclbin / insts / PDI / ELF "
        "into this directory and print where each artifact landed",
    )
    p.add_argument(
        "--from-xclbin",
        type=str,
        default=None,
        help="bring-your-own: run this pre-built xclbin (pairs with --from-insts)",
    )
    p.add_argument(
        "--from-insts",
        type=str,
        default=None,
        help="bring-your-own: instruction binary paired with --from-xclbin",
    )
    return p


def _compile_kwargs(opts):
    return dict(
        problem_size=opts.problem_size,
        aie_tile_width=opts.aie_tile_width,
    )


def aot_compile(opts):
    """Ahead-of-time compile to explicitly named artifacts in ``opts.aot_dir``.

    Demonstrates full control over artifact naming and location: each of the
    xclbin, instruction stream, PDI, and ELF is written to a chosen path
    (cache is bypassed).  ``compile()`` returns ``(xclbin, insts)``; the PDI
    and ELF land at the paths requested.
    """
    out_dir = Path(opts.aot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin_path = out_dir / "vector_scalar_add.xclbin"
    inst_path = out_dir / "vector_scalar_add.insts.bin"
    pdi_path = out_dir / "vector_scalar_add.pdi"
    elf_path = out_dir / "vector_scalar_add.insts.elf"

    device = device_from_args(opts)
    if device is not None:
        iron.set_current_device(device)
    else:
        # No --dev given: bind whatever device the runtime detects, so the
        # generator (which needs an active NPU device) can lower.
        iron.ensure_current_device()
    spec = vector_scalar_add.specialize(**_compile_kwargs(opts))
    spec.compile(
        xclbin_path=xclbin_path,
        inst_path=inst_path,
        pdi_path=pdi_path,
        elf_path=elf_path,
    )
    print("Wrote artifacts:")
    for label, path in (
        ("xclbin", xclbin_path),
        ("insts ", inst_path),
        ("pdi   ", pdi_path),
        ("elf   ", elf_path),
    ):
        print(f"  {label}: {path}  ({'ok' if path.exists() else 'MISSING'})")


def _check_runtime_device(opts):
    """Bind a device and guard the BYO run against an artifact/hardware mismatch.

    Binds the current device (required before ``iron.arange``/``iron.zeros_like``
    can allocate NPU tensors).  When ``--dev`` is given, also checks that the
    attached NPU's architecture matches — a mismatched xclbin typically hangs or
    times out rather than producing clean output, so fail loudly up front.
    """
    # Always probe the runtime first so we know what HW is actually attached.
    runtime_device = iron.ensure_current_device()
    if runtime_device is None:
        raise SystemExit(
            "no NPU runtime device is available to run the pre-built artifacts"
        )

    expected_device = device_from_args(opts)
    if expected_device is not None:
        expected_arch = resolve_target_arch(expected_device)
        runtime_arch = resolve_target_arch(runtime_device)
        if expected_arch != runtime_arch:
            raise SystemExit(
                f"--dev {opts.dev!r} targets {expected_arch}, but the attached NPU "
                f"is {runtime_arch}.  The pre-built artifacts must match the "
                f"runtime device family or the kernel will hang or time out."
            )


def run_from_artifacts(opts):
    """Run a pre-built xclbin + insts, bypassing the @iron.jit generation path.

    ``NPUKernel`` loads any xclbin + instruction-binary pair — it does not care
    whether they came from this design's ``--aot-dir`` export, a Makefile, a
    raw ``aiecc`` invocation, or another tool.  Inputs are passed as IRON
    tensors; the output tensor is pre-allocated and written in place, so its
    element count must match what the artifacts were compiled for
    (``--problem-size``).
    """
    _check_runtime_device(opts)
    npu_kernel = NPUKernel(opts.from_xclbin, opts.from_insts)

    in_t = iron.arange(1, opts.problem_size + 1, dtype=np.int32, device="npu")
    out_t = iron.zeros_like(in_t)

    npu_kernel(in_t, out_t)

    expected = in_t.numpy() + 1
    assert_pass(out_t.numpy(), expected, fail_msg="output does not match in + 1")
    print(
        f"Ran pre-built artifacts:\n  xclbin: {opts.from_xclbin}\n  insts : {opts.from_insts}"
    )


def _run_and_verify(opts):
    in_t = iron.arange(1, opts.problem_size + 1, dtype=np.int32, device="npu")
    out_t = iron.zeros_like(in_t)

    vector_scalar_add(in_t, out_t, **_compile_kwargs(opts))

    expected = in_t.numpy() + 1
    actual = out_t.numpy()
    assert_pass(actual, expected, fail_msg="output does not match in + 1")


def main():
    opts = _make_argparser().parse_args()
    if opts.aot_dir is not None:
        aot_compile(opts)
        return
    if (opts.from_xclbin is None) != (opts.from_insts is None):
        raise SystemExit("--from-xclbin and --from-insts must be set together")
    if opts.from_xclbin is not None:
        run_from_artifacts(opts)
        return
    run_design_cli(
        vector_scalar_add,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
