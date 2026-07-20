# cli.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
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
device binding + ``design.specialize(**kw).compile(
xclbin_path=opts.xclbin_path, inst_path=opts.insts_path
[, elf_path=opts.elf_path][, pdi_path=opts.pdi_path])``.

This module wraps that skeleton so each design just declares the two
pieces it actually owns (compile kwargs, the verify body) and lets the
dispatcher do the branching + the boilerplate around it.  An optional
``emit_mlir=`` callback covers the rare design whose generator needs
real ``iron.tensor`` instances at MLIR-gen time.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypeAlias

if TYPE_CHECKING:
    from aie.iron.device import Device

_DeviceArg: TypeAlias = "Device | Callable[[Any], Device | None]"


def _resolve(value: Any, opts) -> Any:
    """Resolve ``value`` to a concrete value: call it if callable, else pass through."""
    return value(opts) if callable(value) else value


def _resolve_device(value: _DeviceArg, opts) -> "Device":
    """Resolve a CLI device selector to a concrete Device."""
    resolved = value(opts) if callable(value) else value
    if resolved is None:
        raise ValueError("run_design_cli: device selector returned None")
    return resolved


def _runtime_device_name(device: Any) -> str:
    """Return the standard CLI name for a detected NPU device."""
    from aie.utils.compile import resolve_target_arch

    target_arch = resolve_target_arch(device)
    if target_arch == "aie2":
        return "npu"
    if target_arch == "aie2p":
        return "npu2"
    raise RuntimeError(f"Unsupported runtime target architecture: {target_arch}")


def _detect_runtime_target(opts, *, mode: str):
    """Resolve the active NPU family for an automatic CLI target."""
    from aie.utils import get_current_device

    runtime_device = get_current_device()
    if runtime_device is None:
        if mode == "compile":
            sys.exit(
                "run_design_cli: compile-only mode requires --dev when no NPU "
                "runtime device is available."
            )
        sys.exit(
            "run_design_cli: no NPU runtime device is available for an "
            "automatic target selection."
        )

    if hasattr(opts, "dev"):
        opts.dev = _runtime_device_name(runtime_device)
    return runtime_device


def run_design_cli(
    design,
    opts,
    *,
    compile_kwargs: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]],
    run_and_verify: Callable[[Any], None] | None = None,
    device: _DeviceArg | None = None,
    emit_mlir: Callable[[Any], None] | None = None,
    validate: Callable[[Any], None] | None = None,
) -> None:
    """Standard 3-mode CLI dispatcher for basic/ designs.

    The standard branch tree (in order):

      1. Bind an explicit ``--dev`` target or concrete ``device`` argument,
         or detect the attached runtime device for run and local compile-only
         modes.
      2. If ``validate`` is given, call ``validate(opts)``.
      3. If ``opts.emit_mlir`` is True, then:

         * If an ``emit_mlir`` callback was supplied, call ``emit_mlir(opts)``.
         * Otherwise print ``design.specialize(**compile_kwargs).as_mlir()``
           — the right answer for almost every design (no tensor args needed
           when shapes come from ``CompileTime[T]`` params).

         Then return.

      4. If ``opts.xclbin_path`` or ``opts.full_elf_path`` is set (compile-only):

         * For ``full_elf_path``: call ``design.specialize(**compile_kwargs)
           .compile(full_elf_path=opts.full_elf_path)`` — a single
           self-contained ELF, no xclbin/insts.
         * Otherwise refuse if ``opts.insts_path`` is unset (``sys.exit`` with
           the standard message), then call
           ``design.specialize(**compile_kwargs).compile(
           xclbin_path=opts.xclbin_path, inst_path=opts.insts_path,
           [elf_path=opts.elf_path])``.

      5. Otherwise, call ``run_and_verify(opts)``.

    Args:
        design: The ``@iron.jit``-decorated design (a ``CallableDesign``).
        opts: Parsed ``argparse.Namespace`` — must expose at minimum
            ``xclbin_path`` / ``insts_path`` (the standard
            ``add_compile_args`` flags). ``dev`` may be ``None`` to request
            automatic runtime selection. ``emit_mlir`` and ``elf_path`` are
            read if present.
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
            ``opts -> Device``. An explicit ``opts.dev`` is resolved through
            this value, or through ``from_name(opts.dev)`` when omitted. When
            ``opts.dev`` is ``None``, the dispatcher detects the attached NPU
            family, updates ``opts.dev``, and then invokes a callable selector
            so it can retain its declared column profile. Pass a ``Device``
            instance to pin a target independently of the CLI.
        emit_mlir: Optional callable for the ``--emit-mlir`` branch.
            If ``opts.emit_mlir`` is True but this is None, the dispatcher
            falls back to printing
            ``design.specialize(**compile_kwargs).as_mlir()`` — sufficient
            for any design whose generator reads its shapes from
            ``CompileTime[T]`` params (i.e. doesn't read shape off the
            passed-in tensor).  Pass an explicit callable when the
            generator needs real ``iron.tensor`` instances at MLIR-gen
            time.
        validate: Optional callable invoked after target selection — e.g.
            for shape / arg consistency checks that should fire in all modes.
    """
    # Late imports keep this module cheap to import and avoid circular imports
    # until a design actually enters the dispatcher.
    from aie.utils.hostruntime import set_current_device

    emit_mlir_requested = getattr(opts, "emit_mlir", False)
    full_elf_path = getattr(opts, "full_elf_path", None)
    compile_only_requested = (
        getattr(opts, "xclbin_path", None) is not None or full_elf_path is not None
    )
    requested_dev = getattr(opts, "dev", None)

    if getattr(opts, "xclbin_path", None) is not None and not getattr(
        opts, "insts_path", None
    ):
        sys.exit("--xclbin-path requires --insts-path (must be set together)")

    if requested_dev is None:
        has_concrete_device = device is not None and not callable(device)
        if emit_mlir_requested and not has_concrete_device:
            sys.exit(
                "run_design_cli: --emit-mlir requires an explicit target; "
                "pass --dev."
            )

        if device is None and not hasattr(opts, "dev"):
            raise ValueError(
                "run_design_cli: device=None requires opts to expose a "
                "'dev' attribute (the standard add_compile_args flag). "
                "Pass device=<Device or callable> explicitly otherwise."
            )

        if device is not None and not callable(device):
            resolved_device = device
        else:
            mode = "compile" if compile_only_requested else "run"
            _detect_runtime_target(opts, mode=mode)
            if device is None:
                from aie.iron.device import from_name

                resolved_device = from_name(opts.dev)
            else:
                resolved_device = _resolve_device(device, opts)
        set_current_device(resolved_device)
    else:
        if device is None:
            from aie.iron.device import from_name

            resolved_device = from_name(requested_dev)
        else:
            resolved_device = _resolve_device(device, opts)
        set_current_device(resolved_device)

    if validate is not None:
        validate(opts)

    if emit_mlir_requested:
        if emit_mlir is not None:
            emit_mlir(opts)
        else:
            kwargs = _resolve(compile_kwargs, opts)
            print(design.specialize(**kwargs).as_mlir())
        return

    if compile_only_requested:
        kwargs = _resolve(compile_kwargs, opts)
        spec = design.specialize(**kwargs)
        if full_elf_path is not None:
            # Full ELF is self-contained: no xclbin/insts pair.
            spec.compile(full_elf_path=full_elf_path)
            return
        compile_opts = dict(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)
        elf_path = getattr(opts, "elf_path", None)
        if elf_path is not None:
            compile_opts["elf_path"] = elf_path
        pdi_path = getattr(opts, "pdi_path", None)
        if pdi_path is not None:
            compile_opts["pdi_path"] = pdi_path
        spec.compile(**compile_opts)
        return

    if run_and_verify is None:
        sys.exit(
            "run_design_cli: no run_and_verify callback was provided — this "
            "design only supports the compile-only path (pass "
            "--xclbin-path + --insts-path) or --emit-mlir."
        )
    run_and_verify(opts)
