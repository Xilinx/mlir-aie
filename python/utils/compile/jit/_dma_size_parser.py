# _dma_size_parser.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Extract per-host-arg element counts from aiecc's lowered MLIR.

aiecc writes ``input_with_addresses.mlir`` into the kernel directory as part
of compilation.  The host-facing ``aie.runtime_sequence`` carries fully
typed memref arguments — e.g.::

    aie.runtime_sequence @sequence(%arg0: memref<65536xbf16>,
                                   %arg1: memref<65536xbf16>) {
      ...
    }

The argument types ARE the kernel's host-side contract, so we read each
arg's memref shape and compute the element count.  No need to walk
``aie.dma_bd`` ops, distinguish host-facing transfers from tile-internal
DMAs, or fold multi-DMA patterns (fan-out / repeated load / InOut fill+drain)
back together — the runtime_sequence signature already represents what the
caller must supply.

Modules can declare more than one ``aie.runtime_sequence`` (helper
sub-sequences invoked via ``aiex.run @<name>(...)``, or per-device main
sequences in a multi-device program).  We pick the unique **call-graph
root** — the runtime_sequence that no other sequence calls — as the
host-facing entry point.  If there is no unique root (cyclic, or multi-
device with several roots), we return ``None`` and the caller skips
validation rather than risk validating against the wrong signature.

The lowered IR may reference unregistered ops or fail strict verification
(e.g. ``memref.alloca`` outside an ``AutomaticAllocationScope``), so the
parsing context allows unregistered dialects.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_dma_sizes(kernel_dir: Path) -> list[int] | None:
    """Return per-host-arg element counts from ``input_with_addresses.mlir``.

    The returned list is indexed by ``aie.runtime_sequence`` argument
    position.  Each entry is the element count (product of static dims) of
    that arg's memref type.

    Args:
        kernel_dir: Directory aiecc wrote its lowered MLIR into.

    Returns:
        A list of per-arg element counts (length = number of entry-point
        runtime_sequence args), or ``None`` when validation can't be
        performed safely:

        * file is absent or unparseable
        * no runtime_sequence found, or no unique call-graph root (e.g.
          multi-device modules with multiple top-level sequences)
        * any arg has a non-memref type or a dynamic-shape dim
    """
    mlir_path = kernel_dir / "input_with_addresses.mlir"
    if not mlir_path.exists():
        return None
    try:
        # Trigger AIE/aiex dialect registration before constructing the context.
        from aie.dialects import aie as _aie  # noqa: F401
        from aie.dialects import aiex as _aiex  # noqa: F401
        from aie import ir
        from aie._mlir_libs import get_dialect_registry

        ir_context = ir.Context
        ir_location = ir.Location
        ir_module = ir.Module

        ctx = ir_context()
        ctx.append_dialect_registry(get_dialect_registry())
        ctx.load_all_available_dialects()
        ctx.allow_unregistered_dialects = True
        mlir_text = mlir_path.read_text()
        with ctx, ir_location.unknown():
            module = ir_module.parse(mlir_text)

        # Pass 1: collect every runtime_sequence + record aiex.run call edges.
        all_sequences: list = []
        named_sequences: dict = {}  # sym_name -> op  (anonymous ones omitted)
        called: set = set()
        for op in _walk(module.operation):
            if op.name == "aie.runtime_sequence":
                all_sequences.append(op)
                sym = _get_str_attr(op, "sym_name")
                if sym is not None:
                    named_sequences[sym] = op
            elif op.name == "aiex.run":
                target = _get_str_attr(op, "runtime_sequence_symbol")
                if target is not None:
                    called.add(target)

        if not all_sequences:
            return None

        # Pass 2: pick the entry point.
        if len(all_sequences) == 1:
            # Only one sequence in the module — trivially the root.  Works
            # whether or not it carries a sym_name.
            entry = all_sequences[0]
        else:
            # Multiple sequences: each must be named so we can reason about
            # the call graph.  Entry point = the unique root (not called by
            # any other sequence).  Anything else → bail and skip validation.
            if len(named_sequences) != len(all_sequences):
                return None
            roots = [op for sym, op in named_sequences.items() if sym not in called]
            if len(roots) != 1:
                # 0 roots => cyclic; >1 roots => multi-device or multiple
                # top-level entries — can't unambiguously map host tensors.
                return None
            entry = roots[0]

        # Pass 3: read each arg's memref element count.
        seq_block = entry.regions[0].blocks[0]
        sizes: list[int] = []
        memref_type = ir.MemRefType
        for arg in seq_block.arguments:
            t = arg.type
            if not isinstance(t, memref_type):
                return None
            if not t.has_static_shape:
                return None
            elems = 1
            for d in t.shape:
                elems *= d
            sizes.append(elems)
        return sizes if sizes else None
    except Exception:
        # Any binding / parsing failure means validation is unavailable
        # rather than a hard error — a binding regression must not crash
        # the JIT path.  Logged so the regression is still visible.
        logger.debug(
            "parse_dma_sizes: failed to parse %s; tensor validation disabled",
            mlir_path,
            exc_info=True,
        )
        return None


def _walk(op):
    """Yield *op* and every descendant op (pre-order)."""
    yield op
    for region in op.regions:
        for block in region.blocks:
            for sub in block.operations:
                yield from _walk(sub.operation)


def _get_str_attr(op, name: str) -> str | None:
    """Return the string value of attribute *name* on *op*, or ``None``.

    Handles both ``StringAttr`` (``sym_name``) and ``FlatSymbolRefAttr``
    (``aiex.run.runtime_sequence_symbol``); both expose the symbol as
    ``.value``.
    """
    try:
        return op.attributes[name].value
    except (KeyError, AttributeError):
        return None
