# reconfiguration.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``Reconfiguration``: fold several IRON designs into one reconfigurable full ELF.

Folds N designs into one reconfigurable full ELF; each design may bring its
own C++ external kernels.

Usage::

    r = iron.Reconfiguration("prog", method="ctrlpkt", output_dir="out")
    r.add(design_a, inp, out_a)
    r.add(design_b, inp, out_b)
    elf = r.compile()  # -> FullElf

``Reconfiguration`` is compile-only: :meth:`add` stages a design's MLIR text
and kernel objects (retaining neither the design nor the call-time args), and
:meth:`compile` runs ``aiecc --get-full-elf`` once and returns a
:class:`~aie.utils.compile.utils.FullElf` descriptor. It never dispatches.

Every design's ``.mlir`` and every kernel's ``.o`` are staged flat into
``output_dir`` (no per-design subdirectory), and aiecc is run with
``cwd=output_dir``. aiecc resolves each ``link_with="foo.o"`` via
``resolveExternalPath`` (``tools/aiecc/Utils.h``), which tries the process
CWD first, so the flat ``.o`` is found there; ``--tmpdir={name}.prj`` is only
aiecc's private scratch directory, not where the ``.o`` files live.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from .utils import FullElf, _mlir_text_and_name, _run_aiecc

# Recognized --reconfig-method values. Keep in sync with the ReconfigMethod
# cl::values enum (tools/aiecc/CommandLineOptions.h); aiecc is the authority and
# rejects unknown methods itself -- this is a fail-fast mirror so a typo
# surfaces before the fold runs.
_RECONFIG_METHODS = ("loadpdi", "write32", "ctrlpkt")


class Reconfiguration:
    """Builder that folds several designs into one reconfigurable full ELF.

    Args:
        name: Bare identifier used to name the output ELF (``<name>.elf``)
            and aiecc's working directory (``<name>.prj``); must not contain
            a path separator.
        method: ``"loadpdi"``, ``"write32"``, ``"ctrlpkt"``, or ``None``
            (aiecc's default). Only ``"ctrlpkt"`` prepends a ``main:init``
            entrypoint and reserves a control-packet slot.
        output_dir: Directory for the staged inputs, staged kernel objects,
            and the output ELF. Defaults to the current working directory.
            Should be fresh per fold (or have any stale ``.o`` cleared):
            kernel compilation skips rebuilding when its target ``.o``
            already exists, and designs are staged flat into ``output_dir``
            without cleaning it first, so a changed kernel reusing the same
            ``object_file_name`` across folds could otherwise silently link
            from a stale object left by a prior fold.
    """

    def __init__(
        self,
        name: str,
        method: 'Literal["loadpdi", "write32", "ctrlpkt"] | None' = None,
        output_dir: "str | Path | None" = None,
        extra_aiecc_args: "list[str] | None" = None,
    ):
        if "/" in name or "\\" in name:
            raise ValueError(
                f"Reconfiguration: name must be a bare identifier (used to name "
                f"the output ELF and aiecc's working directory), not a path; "
                f"got {name!r}."
            )
        # Fail fast on an unknown method rather than after add()/compile() have
        # staged MLIR and built kernels: aiecc rejects it too, but only late and
        # nested in a subprocess error. None/"" mean "aiecc's default" (matches
        # the `if self._method:` gate in compile()), so let them through.
        if method and method not in _RECONFIG_METHODS:
            raise ValueError(
                f"Reconfiguration: method must be "
                f"{'|'.join(_RECONFIG_METHODS)} (or None for aiecc's default); "
                f"got {method!r}."
            )
        self._name = name
        self._method = method
        self._out = Path(output_dir) if output_dir else Path.cwd()
        # Extra aiecc flags appended to the fold build (e.g. an ablation flag
        # such as --ctrlpkt-pinned-overlay=off). Empty by default, so the
        # ordinary fold is byte-identical.
        self._extra_aiecc_args = list(extra_aiecc_args or [])
        # Each entry: (sym_name, mlir_text, external_kernels, compilable).
        # `compilable` is kept so compile() can build this design's own
        # kernels (via CompilableDesign._build_kernels) without re-generating
        # its MLIR -- _generated_for() memoizes per (device, full_elf, name).
        self._designs: "list[tuple[str, str, list, Any]]" = []

    def add(self, design, *args, **kwargs) -> None:
        """Stage one design's MLIR and external kernels into the fold.

        Args:
            design: A ``CallableDesign`` (an ``@iron.jit``-decorated function).
            *args: Positional tensor args mirroring ``as_mlir()``'s signature.
                Tensor shape/dtype come from the design's own
                ``_TensorPlaceholder``/``CompileTime`` parameters during MLIR
                generation, not from these positional args.
            **kwargs: Any call-time ``CompileTime[T]`` kwargs, the same as you
                would pass to call the design or ``design.as_mlir(...)``.

        Generates this design's MLIR now (needed to discover its
        ``aie.runtime_sequence`` name and its kernel set) but does not build
        or compile anything yet, and does not retain ``args``/``kwargs``
        beyond this call -- :meth:`compile` only replays the design's own
        cached ``(mlir_text, external_kernels)``.
        """
        call_compile_kwargs = design._extract_compile_kwargs(kwargs)[0]
        compilable = design._build_compilable(call_compile_kwargs)
        mlir_text, external_kernels = compilable._generated_for(
            full_elf=compilable.full_elf, reconfig=True
        )
        name = _mlir_text_and_name(mlir_text)[1]
        self._designs.append((name, mlir_text, external_kernels, compilable))

    @staticmethod
    def _trace_buffer_bytes(designs) -> "int | None":
        """Trace-buffer byte count a runlist host must allocate + bind, or None.

        A design that enables hardware trace declares
        ``aie.trace.host_config {buffer_size = N}`` in its runtime_sequence; the
        fold (AIEInsertTraceFlows) then appends a dedicated N-byte trace-buffer
        argument at the tail of that design's own tensor args, before any
        control-packet buffer. Return N so a runlist host can allocate + bind
        that BO (see ``FullElf.trace_buffer_bytes``); None when no design enables
        trace. Only a single-design fold is supported here (the current path
        folds one design at a time); a multi-design fold with trace would need
        per-entrypoint arg accounting, so more than one raises rather than
        silently mis-binding.

        ``designs`` is the staged ``self._designs`` list of
        ``(name, mlir_text, external_kernels, compilable)`` tuples.
        """
        trace_sizes = [
            int(m.group(1))
            for design in designs
            for m in [
                re.search(
                    r"trace\.host_config\s*\{[^}]*buffer_size\s*=\s*(\d+)",
                    design[1],
                )
            ]
            if m
        ]
        if len(trace_sizes) > 1:
            raise RuntimeError(
                "Reconfiguration: hardware trace is only supported when a single "
                "design in the fold enables it; %d designs declare "
                "aie.trace.host_config." % len(trace_sizes)
            )
        return trace_sizes[0] if trace_sizes else None

    def compile(self) -> FullElf:
        """Fold every added design into one reconfigurable full ELF.

        Returns:
            A frozen :class:`FullElf` descriptor: the output path, the
            entrypoints in dispatch order, the ``main:init`` standup entry
            (or ``None``), and whether the fold reserves a control-packet
            slot (both non-``None``/``True`` iff ``method == "ctrlpkt"``).
        """
        if not self._designs:
            raise ValueError(
                "Reconfiguration: no designs added; call add(design, *args) "
                "at least once before compile()."
            )

        names = [n for n, _, _, _ in self._designs]
        if len(set(names)) != len(names):
            raise ValueError(
                "Reconfiguration: duplicate design name; each design needs a "
                "distinct @iron.jit(name=) (the fold hard-fails on duplicate "
                "entrypoints)."
            )

        self._out.mkdir(parents=True, exist_ok=True)

        # Loud same-object-file-name/different-BUILD guard across the WHOLE
        # fold, before any kernel is built. Every design's kernels are staged
        # flat into the same output_dir (see module docstring), so two kernels
        # declaring the same object_file_name whose compiled bytes differ would
        # otherwise silently overwrite one another's `.o`. Key on
        # `_object_content_digest()` (source + flags + includes + toolchain +
        # symbol_prefix), NOT the ExternalFunction *identity* digest: companion
        # kernels from one source file that intentionally share an
        # object_file_name (e.g. `reduce_max_vector` + `compute_max` in
        # `reduce_max.cc`, via `shared_object_file_name`) differ only in
        # name/arg_types and produce a byte-identical `.o`, so they must pass.
        digest_by_object_file: "dict[str, str]" = {}
        for _name, _text, external_kernels, _compilable in self._designs:
            for kernel in external_kernels:
                digest = kernel._object_content_digest()
                prior = digest_by_object_file.get(kernel.object_file_name)
                if prior is not None and prior != digest:
                    raise ValueError(
                        "Reconfiguration: two kernels declare "
                        f"object_file_name={kernel.object_file_name!r} with "
                        "different compiled source; they are staged flat into "
                        "output_dir, so one would silently overwrite the "
                        "other. Give one kernel a distinct "
                        "ExternalFunction(object_file_name=...)."
                    )
                digest_by_object_file[kernel.object_file_name] = digest

        # Stage every design's MLIR flat into output_dir. Basenames only:
        # aiecc is invoked FROM output_dir below.
        staged_names = []
        for name, text, _external_kernels, _compilable in self._designs:
            mlir_path = self._out / f"{name}.mlir"
            mlir_path.write_text(text)
            staged_names.append(mlir_path.name)

        trace_buffer_bytes = self._trace_buffer_bytes(self._designs)

        # Build every design's own kernels FLAT into output_dir (mirrors the
        # 13-matmul non-jit flat build). The guard above already ran, so this
        # cannot silently link the wrong kernel body. Pass reconfig=True to hit
        # the generation `add()` already produced (also reconfig=True); a miss
        # re-runs the generator on a stale ExternalFunction op -> SIGSEGV.
        for _name, _text, _external_kernels, compilable in self._designs:
            compilable._build_kernels(
                str(self._out), full_elf=compilable.full_elf, reconfig=True
            )

        elf_name = f"{self._name}.elf"
        args = [
            "--get-full-elf",
            f"--full-elf-name={elf_name}",
            "--output-dir=.",
            # `--tmpdir` is only aiecc's private scratch subdir; the flat
            # kernel `.o` files are found via aiecc's process CWD (set to
            # output_dir below via cwd=), not via --tmpdir. See the module
            # docstring for the full resolveExternalPath lookup order.
            f"--tmpdir={self._name}.prj",
        ]
        if self._method:
            args.append(f"--reconfig-method={self._method}")
        # Extra aiecc flags for the fold build, from the constructor's
        # extra_aiecc_args (empty by default, so the ordinary fold is unchanged).
        args += self._extra_aiecc_args
        _run_aiecc(staged_names, args, cwd=str(self._out))

        elf_path = self._out / elf_name
        if not elf_path.exists():
            raise RuntimeError(f"Reconfiguration: aiecc produced no ELF at {elf_path}")

        has_init = self._method == "ctrlpkt"
        entrypoints = (["main:init"] if has_init else []) + [f"main:{n}" for n in names]
        return FullElf(
            path=elf_path,
            entrypoints=entrypoints,
            init=("main:init" if has_init else None),
            needs_ctrl_bo=has_init,
            trace_buffer_bytes=trace_buffer_bytes,
        )
