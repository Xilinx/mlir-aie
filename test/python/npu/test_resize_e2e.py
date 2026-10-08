# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Resize images into patches with resize's six entry points, and match resize_ref exactly.

Every core receives every image chunk and both tables down one stream, and
writes its own patch columns. The sizes are chosen for the places a core
can go wrong: pixels straddling the chunk edges, the widest window a chunk
holds, an identity resize of every byte value, and sizes or tables the
kernel refuses, which zero every patch, one core's, or every band from one
on, while the streams stay in step. One build also resizes several images
in turn, as a per-call size runs it, so no state outlives its image.
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
from ml_dtypes import bfloat16

SIDE = 16
PATCH = SIDE * SIDE * 3


@jit
def resize_patches(
    image_in: In,
    taps_in: In,
    patches_out: Out,
    *,
    sizes: CompileTime[tuple],
    words: CompileTime[int] = 356,
    chunk: CompileTime[int] = 4096,
    patch_columns: CompileTime[int] = 8,
    cores: CompileTime[int] = 2,
):
    """Each of ``sizes``' images in turn: its chunks and its width then height table in, each core's patches out."""
    fn = vision.resize(words, chunk=chunk, patch_columns=patch_columns, cores=cores)
    counts_ty = np.ndarray[(5,), np.dtype[np.int32]]
    image_ty = np.ndarray[(chunk,), np.dtype[np.uint8]]
    taps_ty = np.ndarray[(words,), np.dtype[np.int32]]
    patch_ty = np.ndarray[(PATCH,), np.dtype[bfloat16]]
    of_image = ObjectFifo(image_ty, name="image", depth=2)
    of_taps = ObjectFifo(taps_ty, name="taps", depth=2)
    of_patches = [
        ObjectFifo(patch_ty, name=f"patches_{k}", depth=2) for k in range(cores)
    ]

    def core_body(
        core, image, taps, patches, counts, consume, setup, take, band, emit, finish
    ):
        for rows, columns, out_rows, out_columns in sizes:
            setup(counts, rows, columns, out_rows, out_columns, core)
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
                    patch = patches.acquire(1)
                    emit(patch, i)
                    patches.release(1)
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
                of_patches[k].prod(),
                Buffer(counts_ty, name=f"counts_{k}"),
                fn,
                fn.resize_setup,
                fn.resize_take,
                fn.resize_band,
                fn.resize_emit,
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
        np.ndarray[(patch_rows * PATCH,), np.dtype[bfloat16]],
    ]

    def sequence(image_h, taps_h, patches_h, image, taps, *outs):
        # Every drain in flight at once: a core whose patches wait holds up
        # the broadcast image for the rest.
        tg = TaskGroup()
        image.fill(image_h, group=tg)
        taps.fill(taps_h, group=tg)
        # Core k's patches are rows k, k + cores, ... of every image's bands,
        # each a whole number of rows of cores.
        for k, out in enumerate(outs):
            mine = TensorAccessPattern(
                (patch_rows, PATCH),
                k * PATCH,
                [1, 1, patch_rows // cores, PATCH],
                [0, 0, cores * PATCH, 1],
            )
            out.drain(patches_h, tap=mine, wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [*host, of_image.prod(), of_taps.prod(), *[f.cons() for f in of_patches]],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _table(in_size, out_size, words):
    """A 16-slot table of ``in_size`` samples to ``out_size``, one chunk a patch."""
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


def _image(rows, columns, out_rows, out_columns, tables, expect, **build):
    """One image's inputs and its expected patches, checked to say what ``expect`` does."""
    words, chunk, cores = build["words"], build["chunk"], build["cores"]
    width_in, height_in = tables or (None, None)
    strips, bands = out_columns // SIDE, out_rows // SIDE
    taps_w = _table(width_in or columns, out_columns, words)[:strips]
    taps_h = _table(height_in or rows, out_rows, words)[:bands]
    cpr = -(-3 * columns // chunk)
    rng = np.random.default_rng(rows * columns + chunk)
    image = rng.integers(0, 256, (rows, cpr * chunk), dtype=np.uint8)
    expected = vision.resize_ref(
        image, taps_w, taps_h, rows, columns, out_rows, out_columns, **build
    )

    pad = expected.shape[0] // bands
    live = (expected.reshape(bands, pad, -1).view(np.uint16) != 0).any(axis=-1)
    patch = live[:, :strips]
    assert not live[:, strips:].any()
    if expect in ("all", "identity"):
        assert patch.all()
    elif expect == "none":
        assert not patch.any()
    elif expect == "some cores":
        per_core = [patch[:, k::cores].any() for k in range(cores)]
        assert any(per_core) and not all(per_core)
    else:
        per_band = patch.any(axis=1)
        assert per_band[0] and not per_band[-1]
    if expect == "identity":
        assert np.unique(expected.view(np.uint16)).size == 256
    return image, np.concatenate([taps_w, taps_h]), expected


@pytest.mark.parametrize(
    "images,cores,chunk,patch_columns",
    [
        # Up, every pixel row in 64-byte chunks: pixels straddle their edges.
        ([(37, 53, 96, 144, None, "all")], 2, 64, 8),
        # Down 9.375 on both axes: a 39-tap window, the most 356 words hold.
        ([(1500, 1200, 160, 128, None, "all")], 2, 4096, 8),
        # Down on 4 cores, a core with one patch column fewer than the rest.
        ([(300, 400, 64, 96, None, "all")], 4, 4096, 8),
        # Identity over every byte value: the bf16 of each u8 / 255.
        ([(32, 48, 32, 48, None, "identity")], 2, 4096, 8),
        # Sizes refused: 100 rows out are no whole band.
        ([(37, 53, 100, 144, None, "none")], 2, 4096, 8),
        # 9 patch columns over 2 cores is 5 a core; a core holds 2.
        ([(37, 53, 96, 144, None, "none")], 2, 4096, 2),
        # A width table for 56 columns: patch column 8 reads past 53, and
        # its core (0 of 4) writes zeros.
        ([(37, 53, 96, 144, (56, None), "some cores")], 4, 4096, 8),
        # A height table for 60 rows: the bands past the 37th row.
        ([(37, 53, 96, 144, (None, 60), "some bands")], 2, 4096, 8),
        # One build, three images in turn, as a per-call size runs it: fewer
        # patch columns than the first, then sizes refused, each leaving
        # nothing of the one before.
        (
            [
                (37, 53, 96, 144, None, "all"),
                (37, 53, 96, 32, None, "all"),
                (37, 53, 100, 144, None, "none"),
            ],
            2,
            64,
            8,
        ),
    ],
)
def test_device_patches_match_resize_ref(images, cores, chunk, patch_columns):
    build = dict(words=356, chunk=chunk, patch_columns=patch_columns, cores=cores)
    parts = [_image(*spec, **build) for spec in images]
    expected = np.concatenate([e for _, _, e in parts])
    patches = iron.full((expected.size,), 1.0, dtype=bfloat16)
    resize_patches(
        iron.tensor(
            np.concatenate([i.reshape(-1) for i, _, _ in parts]), dtype=np.uint8
        ),
        iron.tensor(
            np.concatenate([t.reshape(-1) for _, t, _ in parts]), dtype=np.int32
        ),
        patches,
        sizes=tuple(spec[:4] for spec in images),
        **build,
    )
    got = patches.numpy().reshape(expected.shape)
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
