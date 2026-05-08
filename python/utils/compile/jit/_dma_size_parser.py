# _dma_size_parser.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Extract per-transfer element counts from aiecc's lowered MLIR.

aiecc writes ``input_with_addresses.mlir`` into the kernel directory as part
of compilation.  The host-facing ``aie.runtime_sequence`` block contains one
``aie.dma_bd`` op per per-column DMA transfer; the element count rides on
the op's ``len`` attribute.

We parse the file with the AIE MLIR Python bindings rather than regex so the
extractor is not coupled to the textual custom-assembly form — we read the
``len`` attribute and the operand structure directly.

Only ``aie.dma_bd`` ops whose first operand is a block argument of the
enclosing ``aie.runtime_sequence`` are counted; tile-internal DMAs that
reference named buffer SSA values are excluded.

The lowered IR may reference unregistered ops or fail strict verification
(e.g. ``memref.alloca`` outside an ``AutomaticAllocationScope``), so the
parsing context allows unregistered dialects.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_dma_sizes(kernel_dir: Path) -> list[int] | None:
    """Return per-transfer element counts from ``input_with_addresses.mlir``.

    Args:
        kernel_dir: Directory aiecc wrote its lowered MLIR into.

    Returns:
        A list of element counts in transfer order, or ``None`` when the
        file is absent or unparseable.
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

        ctx = ir.Context()
        ctx.append_dialect_registry(get_dialect_registry())
        ctx.load_all_available_dialects()
        ctx.allow_unregistered_dialects = True
        with ctx, ir.Location.unknown():
            module = ir.Module.parse(mlir_path.read_text())

        seq = _find_runtime_sequence(module.operation)
        if seq is None:
            return None
        seq_block = seq.regions[0].blocks[0]

        sizes: list[int] = []
        for op in _walk(seq):
            if op.name != "aie.dma_bd" or len(op.operands) == 0:
                continue
            # First operand is the memref being transferred.  When it owns to
            # the runtime_sequence's own block, it's a host-facing %argN
            # rather than a tile-internal named buffer.
            if op.operands[0].owner == seq_block:
                sizes.append(int(op.attributes["len"].value))
        return sizes or None
    except Exception:
        # Treat any binding/parsing failure as "validation unavailable" rather
        # than crashing the caller — runtime tensor validation is best-effort.
        # Logged so a binding regression doesn't silently disable validation.
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


def _find_runtime_sequence(op):
    """Return the first ``aie.runtime_sequence`` op found, or ``None``."""
    for descendant in _walk(op):
        if descendant.name == "aie.runtime_sequence":
            return descendant
    return None
