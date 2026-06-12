# cli.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""``run_design_cli`` — standard 3-mode dispatcher for IRON design CLIs.

Almost every basic/ design's ``main()`` is the same skeleton:

    def main():
        opts = _make_argparser().parse_args()
        # optional: _validate(opts)
        if opts.emit_mlir:
            print(design.specialize(**_compile_kwargs(opts)).as_mlir()); return
        if opts.xclbin_path:
            _compile_only(opts)
            return
        _run_and_verify(opts)

…where ``_compile_only`` always does the same ``--insts-path`` check +
``set_current_device(from_name(opts.dev))`` + ``design.specialize(**kw)
.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path
[, elf_path=opts.elf_path])``.

This module wraps that skeleton so each design just declares the two
pieces it actually owns (compile kwargs, the verify body) and lets the
dispatcher do the branching + the boilerplate around it.  An optional
``emit_mlir=`` callback covers the rare design whose generator needs
real ``iron.tensor`` instances at MLIR-gen time.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Mapping


def _resolve(value: Any, opts) -> Any:
    """Resolve ``value`` to a concrete value: call it if callable, else pass through."""
    return value(opts) if callable(value) else value


def run_design_cli(
    design,
    opts,
    *,
    compile_kwargs: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]],
    run_and_verify: Callable[[Any], None] | None = None,
    device: Any | Callable[[Any], Any] | None = None,
    emit_mlir: Callable[[Any], None] | None = None,
    validate: Callable[[Any], None] | None = None,
) -> None:
    """Standard 3-mode CLI dispatcher for basic/ designs.

    The standard branch tree (in order):

      1. If ``validate`` is given, call ``validate(opts)`` first.
      2. If ``opts.emit_mlir`` is True, set the current device, then:

         * If an ``emit_mlir`` callback was supplied, call ``emit_mlir(opts)``.
         * Otherwise print ``design.specialize(**compile_kwargs).as_mlir()``
           — the right answer for almost every design (no tensor args needed
           when shapes come from ``CompileTime[T]`` params).

         Then return.

      3. If ``opts.xclbin_path`` is set:

         * Refuse if ``opts.insts_path`` is unset (``sys.exit`` with the
           standard message).
         * Set the current device.
         * Call ``design.specialize(**compile_kwargs).compile(
           xclbin_path=opts.xclbin_path, inst_path=opts.insts_path,
           [elf_path=opts.elf_path])``.

      4. Otherwise, call ``run_and_verify(opts)``.

    Args:
        design: The ``@iron.jit``-decorated design (a ``CallableDesign``).
        opts: Parsed ``argparse.Namespace`` — must expose at minimum
            ``xclbin_path`` / ``insts_path`` (the standard
            ``add_compile_args`` flags).  ``emit_mlir`` and ``elf_path``
            are read if present.
        compile_kwargs: Either a dict OR a callable that takes ``opts``
            and returns the kwargs dict to pass to
            ``design.specialize()``.  Callable form is convenient for
            the typical ``_compile_kwargs(opts)`` helper most designs
            already have.
        run_and_verify: Callable invoked in the default (NPU
            run + numpy verify) branch.  Takes ``opts``, returns nothing
            — exits non-zero on failure (e.g. via ``assert_pass``).
            Optional: ml/ designs whose verification lives in a C++ test
            harness omit this; reaching the run branch without it exits
            with a clear "no python run path" message.
        device: Optional iron ``Device`` instance OR callable
            ``opts -> Device``.  If omitted, defaults to
            ``from_name(opts.dev)`` (using whatever device choices the
            argparse setup allowed).  Pass a callable when the device
            depends on more than just ``opts.dev`` — e.g. matmul's
            ``(opts.dev, opts.n_aie_cols)`` mapping.
        emit_mlir: Optional callable for the ``--emit-mlir`` branch.
            If ``opts.emit_mlir`` is True but this is None, the dispatcher
            falls back to printing
            ``design.specialize(**compile_kwargs).as_mlir()`` — sufficient
            for any design whose generator reads its shapes from
            ``CompileTime[T]`` params (i.e. doesn't read shape off the
            passed-in tensor).  Pass an explicit callable when the
            generator needs real ``iron.tensor`` instances at MLIR-gen
            time.
        validate: Optional callable invoked before any branch — e.g. for
            shape / arg consistency checks that should fire in all modes.
    """
    # Late imports so this module is cheap to import even when no
    # design ever calls it (and to dodge circular-import issues with
    # aie.iron.device).
    from aie.iron.device import from_name
    from aie.utils.hostruntime import set_current_device

    if validate is not None:
        validate(opts)

    if device is None:
        # Default: read opts.dev and pass through from_name.
        if not hasattr(opts, "dev"):
            raise ValueError(
                "run_design_cli: device=None requires opts to expose a "
                "'dev' attribute (the standard add_compile_args flag). "
                "Pass device=<Device or callable> explicitly otherwise."
            )

        def _default_device(opts):
            return from_name(opts.dev)

        device = _default_device

    if getattr(opts, "emit_mlir", False):
        set_current_device(_resolve(device, opts))
        if emit_mlir is not None:
            emit_mlir(opts)
        else:
            kwargs = _resolve(compile_kwargs, opts)
            print(design.specialize(**kwargs).as_mlir())
        return

    if getattr(opts, "xclbin_path", None):
        if not getattr(opts, "insts_path", None):
            sys.exit("--xclbin-path requires --insts-path (must be set together)")
        set_current_device(_resolve(device, opts))
        kwargs = _resolve(compile_kwargs, opts)
        spec = design.specialize(**kwargs)
        compile_opts = dict(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)
        elf_path = getattr(opts, "elf_path", None)
        if elf_path is not None:
            compile_opts["elf_path"] = elf_path
        spec.compile(**compile_opts)
        return

    if run_and_verify is None:
        sys.exit(
            "run_design_cli: no run_and_verify callback was provided — this "
            "design only supports the compile-only path (pass "
            "--xclbin-path + --insts-path) or --emit-mlir."
        )
    run_and_verify(opts)
