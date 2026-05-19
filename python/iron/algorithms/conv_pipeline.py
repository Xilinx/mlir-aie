# conv_pipeline.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Control-flow templates for tile-pipeline conv worker functions.

These helpers are invoked from inside a Worker function body. They emit the
acquire/release plumbing for common row-driver and 3-row sliding-window
patterns; the caller passes a ``do_kernel`` closure that supplies the
design-specific kernel call.

Two weight-delivery modes are supported:

  * ``of_wts=None`` — caller closes over a static ``Buffer`` (always
    available on the tile, no acquire/release).
  * ``of_wts=ObjectFifo`` — weights arrive via an ObjectFifo; the helper
    acquires once outside the row loop and releases at the end. The
    acquired view is passed through to ``do_kernel`` as its last positional
    argument.

These are deliberately small — they capture the boilerplate that mobilenet's
``build_3tile_pipeline`` and resnet's ``conv1_fn`` / ``conv2_fn`` /
``conv1_skip_fn`` both reinvent. Larger orchestration (placement, fifo
declarations, weight delivery choice) stays at the call site.
"""

from aie.iron.controlflow import range_


def row_at_a_time(of_in, of_out, *, n_rows, do_kernel, of_wts=None):
    """1x1-style row driver.

    For each of ``n_rows`` rows: acquire 1 input row, acquire 1 output row,
    call ``do_kernel(r_in, r_out, wts_view_or_None)``, release both.

    If ``of_wts`` is given, acquire 1 weights view once outside the loop and
    release once at the end. Otherwise, ``None`` is passed in the wts slot.
    """
    wts = of_wts.acquire(1) if of_wts is not None else None
    for _ in range_(n_rows):
        r_in = of_in.acquire(1)
        r_out = of_out.acquire(1)
        do_kernel(r_in, r_out, wts)
        of_in.release(1)
        of_out.release(1)
    if of_wts is not None:
        of_wts.release(1)


def sliding_3row(of_in, of_out, *, n_out_rows, do_kernel, of_wts=None):
    """3-row sliding window with replicate-border phasing.

    Emits the standard preamble / middle / postamble dance for a 3-tap
    vertical convolution that replicates the boundary rows:

      * preamble (1 output row): acquire 2 input rows, call with
        ``(rows[0], rows[0], rows[1])`` and ``border=0``; no input release.
      * middle (``n_out_rows - 2`` output rows): acquire 3 input rows, call
        with ``(rows[0], rows[1], rows[2])`` and ``border=1``; release 1
        input row.
      * postamble (1 output row): acquire 2 input rows, call with
        ``(rows[0], rows[1], rows[1])`` and ``border=2``; release 2 input
        rows.

    ``do_kernel(top, mid, bot, r_out, wts_view_or_None, border)``.
    """
    wts = of_wts.acquire(1) if of_wts is not None else None

    # preamble
    rows = of_in.acquire(2)
    r_out = of_out.acquire(1)
    do_kernel(rows[0], rows[0], rows[1], r_out, wts, 0)
    of_out.release(1)

    # middle
    for _ in range_(n_out_rows - 2):
        rows = of_in.acquire(3)
        r_out = of_out.acquire(1)
        do_kernel(rows[0], rows[1], rows[2], r_out, wts, 1)
        of_in.release(1)
        of_out.release(1)

    # postamble
    rows = of_in.acquire(2)
    r_out = of_out.acquire(1)
    do_kernel(rows[0], rows[1], rows[1], r_out, wts, 2)
    of_in.release(2)
    of_out.release(1)

    if of_wts is not None:
        of_wts.release(1)


def row_at_a_time_with_skip(of_in, of_out, of_skip, *, n_rows, do_kernel, of_wts=None):
    """row_at_a_time variant with a per-row skip fifo.

    For each row: acquire 1 input, 1 output, 1 skip; call
    ``do_kernel(r_in, r_out, r_skip, wts_view_or_None)``; release all three.
    """
    wts = of_wts.acquire(1) if of_wts is not None else None
    for _ in range_(n_rows):
        r_in = of_in.acquire(1)
        r_out = of_out.acquire(1)
        r_skip = of_skip.acquire(1)
        do_kernel(r_in, r_out, r_skip, wts)
        of_in.release(1)
        of_out.release(1)
        of_skip.release(1)
    if of_wts is not None:
        of_wts.release(1)


def row_at_a_time_tiled(
    of_in, of_out, *, n_rows, n_tiles_per_row, do_kernel, of_wts=None
):
    """Row driver that splits each output row into ``n_tiles_per_row`` tiles.

    For each of ``n_rows`` rows, and for each of ``n_tiles_per_row`` tiles
    within a row: acquire 1 input tile and 1 output tile, call
    ``do_kernel(r_in, r_out, wts_view_or_None)``, release both.

    Equivalent to nesting ``row_at_a_time`` inside an outer row loop, but
    more economical for kernels that already do their own spatial-tile
    indexing per call.
    """
    wts = of_wts.acquire(1) if of_wts is not None else None
    for _ in range_(n_rows):
        for _ in range_(n_tiles_per_row):
            r_in = of_in.acquire(1)
            r_out = of_out.acquire(1)
            do_kernel(r_in, r_out, wts)
            of_in.release(1)
            of_out.release(1)
    if of_wts is not None:
        of_wts.release(1)
