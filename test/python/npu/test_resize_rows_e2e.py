# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""resize's resize_emit_row, which test_resize_e2e.py's state machine does
not exercise: the same setup/take/band/consume/finish, but reading a band's
patch columns as interleaved uint8 RGB rows rather than bf16 u8/255
patches, matched against resize_ref(as_rows=True) exactly. The validity and
zero-padding rules are resize_emit's (test_resize_e2e.py), so this covers
only the few places the row path differs: an image row, a zeroed band, and
one core short a patch column.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.iron.kernels import vision

SIDE = 16
ROW_BYTES = SIDE * SIDE * 3


@jit
def resize_rows(
    image_in: In,
    taps_in: In,
    rows_out: Out,
    *,
    sizes: CompileTime[tuple],
    words: CompileTime[int] = 356,
    chunk: CompileTime[int] = 4096,
    patch_columns: CompileTime[int] = 8,
    cores: CompileTime[int] = 2,
):
    """Each of ``sizes``' images in turn: its chunks and its width then height table in, each core's rows out."""
    fn = vision.resize(words, chunk=chunk, patch_columns=patch_columns, cores=cores)
    counts_ty = np.ndarray[(5,), np.dtype[np.int32]]
    image_ty = np.ndarray[(chunk,), np.dtype[np.uint8]]
    taps_ty = np.ndarray[(words,), np.dtype[np.int32]]
    row_ty = np.ndarray[(ROW_BYTES,), np.dtype[np.uint8]]
    of_image = ObjectFifo(image_ty, name="image", depth=2)
    of_taps = ObjectFifo(taps_ty, name="taps", depth=2)
    of_rows = [ObjectFifo(row_ty, name=f"rows_{k}", depth=2) for k in range(cores)]

    def core_body(
        core, image, taps, out_rows, counts, consume, setup, take, band, emit, finish
    ):
        for rows, columns, out_h, out_w in sizes:
            setup(counts, rows, columns, out_h, out_w, core)
            for k in range_(counts[0]):
                width = taps.acquire(1)
                take(width, k)
                taps.release(1)
            for _ in range_(counts[1]):
                height = taps.acquire(1)
                band(height, counts)
                for _ in range_(counts[2]):
                    pixels = image.acquire(1)
                    consume(pixels, height)
                    image.release(1)
                for i in range_(counts[3]):
                    row = out_rows.acquire(1)
                    emit(row, i)
                    out_rows.release(1)
                taps.release(1)
            finish(counts)
            for _ in range_(counts[4]):
                image.acquire(1)
                image.release(1)

    workers = [
        Worker(
            core_body,
            [
                k,
                of_image.cons(),
                of_taps.cons(),
                of_rows[k].prod(),
                Buffer(counts_ty, name=f"counts_{k}"),
                fn,
                fn.resize_setup,
                fn.resize_take,
                fn.resize_band,
                fn.resize_emit_row,
                fn.resize_finish,
            ],
        )
        for k in range(cores)
    ]

    image_bytes = taps_words = patch_rows = 0
    for rows, columns, out_rows, out_columns in sizes:
        strips, bands = out_columns // SIDE, out_rows // SIDE
        image_bytes += rows * -(-3 * columns // chunk) * chunk
        taps_words += (strips + bands) * words
        patch_rows += bands * (strips // cores + 1) * cores
    host = [
        np.ndarray[(image_bytes,), np.dtype[np.uint8]],
        np.ndarray[(taps_words,), np.dtype[np.int32]],
        np.ndarray[(patch_rows * ROW_BYTES,), np.dtype[np.uint8]],
    ]

    def sequence(image_h, taps_h, rows_h, image, taps, *outs):
        tg = TaskGroup()
        image.fill(image_h, group=tg)
        taps.fill(taps_h, group=tg)
        for k, out in enumerate(outs):
            mine = TensorAccessPattern(
                (patch_rows, ROW_BYTES),
                k * ROW_BYTES,
                [1, 1, patch_rows // cores, ROW_BYTES],
                [0, 0, cores * ROW_BYTES, 1],
            )
            out.drain(rows_h, tap=mine, wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [*host, of_image.prod(), of_taps.prod(), *[f.cons() for f in of_rows]],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _table(in_size, out_size, words):
    """A 16-slot table of ``in_size`` samples to ``out_size``, one chunk a patch column."""
    chunks = -(-out_size // SIDE)
    peak = vision.resample_peak_ref(
        in_size, out_size, 0, chunks, words=words, cores=1, slots=SIDE
    )
    return vision.resample_quantize_ref(
        np.tile(peak, (chunks, 1)),
        in_size,
        out_size,
        0,
        chunks,
        words=words,
        cores=1,
        slots=SIDE,
    )


def _image(rows, columns, out_rows, out_columns, **build):
    """One image's inputs and its expected rows (resize_ref(as_rows=True)), as
    patch-column-shaped slices in the same layout the device streams write."""
    words, chunk, cores = build["words"], build["chunk"], build["cores"]
    strips, bands = out_columns // SIDE, out_rows // SIDE
    taps_w = _table(columns, out_columns, words)[:strips]
    taps_h = _table(rows, out_rows, words)[:bands]
    cpr = -(-3 * columns // chunk)
    rng = np.random.default_rng(rows * columns + chunk)
    image = rng.integers(0, 256, (rows, cpr * chunk), dtype=np.uint8)
    raster = vision.resize_ref(
        image,
        taps_w,
        taps_h,
        rows,
        columns,
        out_rows,
        out_columns,
        as_rows=True,
        **build,
    )
    pad = (strips // cores + 1) * cores
    expected = np.zeros((bands, pad, SIDE, SIDE, 3), np.uint8)
    expected[:, :strips] = raster.reshape(bands, SIDE, strips, SIDE, 3).transpose(
        0, 2, 1, 3, 4
    )
    return image, np.concatenate([taps_w, taps_h]), expected.reshape(bands * pad, -1)


@pytest.mark.parametrize(
    "images,cores,chunk,patch_columns",
    [
        # Down on 2 cores: a plain case, every row the image's.
        ([(300, 400, 64, 96)], 2, 4096, 8),
        # Up, every pixel row in 64-byte chunks: pixels straddle their edges.
        ([(37, 53, 96, 144)], 2, 64, 8),
        # 9 patch columns over 4 cores, core 0 holding all 3 it can.
        ([(37, 53, 96, 144)], 4, 4096, 3),
        # One build, two images in turn, as a per-call size runs it.
        ([(37, 53, 96, 144), (37, 53, 96, 32)], 2, 64, 8),
    ],
)
def test_device_rows_match_resize_ref(images, cores, chunk, patch_columns):
    build = dict(words=356, chunk=chunk, patch_columns=patch_columns, cores=cores)
    parts = [_image(*spec, **build) for spec in images]
    expected = np.concatenate([e for _, _, e in parts])
    rows = iron.full((expected.size,), 1, dtype=np.uint8)
    resize_rows(
        iron.tensor(
            np.concatenate([i.reshape(-1) for i, _, _ in parts]), dtype=np.uint8
        ),
        iron.tensor(
            np.concatenate([t.reshape(-1) for _, t, _ in parts]), dtype=np.int32
        ),
        rows,
        sizes=tuple(spec[:4] for spec in images),
        **build,
    )
    got = rows.numpy().reshape(expected.shape)
    np.testing.assert_array_equal(got, expected)
