#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Per-block standalone IRON design: build + emit MLIR for ONE bottleneck.

Usage:
    python3 aie2_iron_per_block.py <block_name>   > /tmp/<block_name>.mlir
    python3 aie2_iron_per_block.py <block_name> --xclbin-path x.xclbin --insts-path i.bin

Examples:
    python3 aie2_iron_per_block.py bn3            # regular (single tile)
    python3 aie2_iron_per_block.py bn4_5          # fused pair on one tile
    python3 aie2_iron_per_block.py bn11           # 3-tile pipeline (with skip)
    python3 aie2_iron_per_block.py bn12           # 2-tile fused
    python3 aie2_iron_per_block.py bn13           # cascade (5 tiles + 3 memtiles)

Useful for isolating one block to debug or measure independently — the IRON
analogue of bottleneck_A/test_bn_*.py. The block is wrapped in a minimal
shim-fill -> block -> shim-drain skeleton; the SA placer picks the tiles.
Reuses the lifted module-level builders from bottleneck/{regular,pipeline,
cascade}.py — same builder, different runtime wiring.
"""

import argparse
import json
import os

import aie.iron as iron
import numpy as np
from aie.iron import (
    CompileTime,
    InOut,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
)
from aie.utils.hostruntime import set_current_device
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args

from .bottleneck._common import i8 as _i8
from .bottleneck._common import sa_placer_flags
from .bottleneck._common import u8 as _u8
from .bottleneck.cascade import build_cascade
from .bottleneck.pipeline import build_3tile_pipeline, build_bn12_2tile
from .bottleneck.regular import build_2layer_skip, build_3layer, build_fused_pair
from .network_spec import CASCADE_NAMES
from .network_spec import block as nsblock

DATA_DIR = os.path.join(os.path.dirname(__file__), "data") + "/"
SCALE_FACTORS = None  # Lazy-loaded in per_block_iron from --scales-json or default.


def _resolve_scales(scales_json_path):
    """Load the scale_factors JSON. Defaults to the main mobilenet calibration."""
    if scales_json_path is None:
        scales_json_path = DATA_DIR + "scale_factors_final.json"
    with open(scales_json_path) as f:
        return json.load(f)


# Block-name → builder dispatch.
_FUSED_PAIRS = {"bn4_5": ("bn4", "bn5"), "bn8_9": ("bn8", "bn9")}

# Set by per_block_iron(); referenced by _build_one to control which weight
# files and scale factors get baked into the design.
_DATA_DIR = DATA_DIR
_SCALES = None
_WTS_TAG = "chain"


def _build_one(block_name, act_in):
    """Dispatch to the right builder. Returns (out_fifo, workers, weight_fifos)."""
    if block_name == "bn0":
        out_fifo, w = build_2layer_skip(
            nsblock("bn0"),
            act_in,
            _SCALES,
            data_dir=_DATA_DIR,
            wts_tag=_WTS_TAG,
        )
        return out_fifo, [w], []

    if block_name in ("bn1", "bn2", "bn3", "bn6", "bn7", "bn8"):
        # bn8 isn't a standalone block in the main mobilenet (it lives in the
        # bn8_9 fused pair), but its 3-layer shape works as a standalone test
        # block — useful for matching the bottleneck_A bn8 brevitas fixture.
        out_fifo, w = build_3layer(
            nsblock(block_name),
            act_in,
            _SCALES,
            data_dir=_DATA_DIR,
            wts_tag=_WTS_TAG,
        )
        return out_fifo, [w], []

    if block_name in _FUSED_PAIRS:
        a, b = _FUSED_PAIRS[block_name]
        chain = f"{a}_{b[2:]}_chain.txt"
        out_fifo, w = build_fused_pair(
            nsblock(a),
            nsblock(b),
            chain,
            act_in,
            _SCALES,
            data_dir=_DATA_DIR,
        )
        return out_fifo, [w], []

    if block_name == "bn10":
        out_fifo, ws = build_3tile_pipeline(
            nsblock("bn10"),
            act_in,
            _SCALES,
            data_dir=_DATA_DIR,
        )
        return out_fifo, ws, []

    if block_name == "bn11":
        # bn11 has a skip path forwarded through a memtile.
        skip_in = act_in.cons(depth=6).forward(depth=2)
        out_fifo, ws = build_3tile_pipeline(
            nsblock("bn11"),
            act_in,
            _SCALES,
            data_dir=_DATA_DIR,
            skip_in=skip_in,
        )
        return out_fifo, ws, []

    if block_name == "bn12":
        out_fifo, ws = build_bn12_2tile(
            nsblock("bn12"),
            act_in,
            _SCALES,
            data_dir=_DATA_DIR,
        )
        return out_fifo, ws, []

    if block_name in CASCADE_NAMES:
        out_fifo, wts_l1, wts_l3, ws = build_cascade(
            nsblock(block_name),
            act_in=act_in,
            skip_in=act_in,
            sf=_SCALES,
            data_dir=_DATA_DIR,
        )
        return out_fifo, ws, [wts_l1, wts_l3]

    raise ValueError(
        f"unsupported block name: {block_name!r} "
        f"(supported: bn0..bn3, bn4_5, bn6, bn7, bn8_9, bn10..bn14)"
    )


def per_block_iron(
    block_name: CompileTime[str],
    data_dir: CompileTime[str | None] = None,
    scales_json: CompileTime[str | None] = None,
    wts_tag: CompileTime[str] = "chain",
):
    """Build a standalone IRON design for one bottleneck and return MLIR.

    data_dir / scales_json default to the main mobilenet calibration; pass the
    bottleneck_A|B|C/data/ path + matching scale_factors.json to build a design
    that targets the per-bn brevitas fixtures. Single-tile blocks load weights
    from <bn>_<wts_tag>.txt; bottleneck_A keeps its per-bn weights as "single".
    """
    global _DATA_DIR, _SCALES, _WTS_TAG
    _WTS_TAG = wts_tag
    _DATA_DIR = (
        data_dir + "/"
        if data_dir and not data_dir.endswith("/")
        else (data_dir or DATA_DIR)
    )
    _SCALES = _resolve_scales(scales_json)
    blk = (
        nsblock(block_name)
        if block_name not in _FUSED_PAIRS
        else nsblock(_FUSED_PAIRS[block_name][0])
    )
    in_w, in_h, in_c = blk.layers[0].in_shape
    out_blk = (
        nsblock(_FUSED_PAIRS[block_name][1]) if block_name in _FUSED_PAIRS else blk
    )
    out_w, out_h, out_c = out_blk.layers[-1].out_shape

    # i32-flat host buffer types (size in bytes / 4).
    in_byte_count = in_w * in_h * in_c
    out_byte_count = out_w * out_h * out_c
    in_ty = np.ndarray[(in_byte_count // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[(out_byte_count // 4,), np.dtype[np.int32]]

    # bn0 takes uint8 (init-conv output); everything else takes int8.
    in_elem_ty = _u8 if block_name == "bn0" else _i8
    act_in = ObjectFifo(in_elem_ty((in_w, 1, in_c)), depth=2)

    out_fifo, workers, wts_fifos = _build_one(block_name, act_in)

    if wts_fifos:
        # Cascade: input + 2 weight buffers + output.
        BN_WTS_SZ = 80 * 960  # 76800 bytes per L1/L3 weight chunk for bn13/bn14
        wts_ty = np.ndarray[(BN_WTS_SZ // 4,), np.dtype[np.int32]]

        def sequence_with_wts(
            inp, wl1, wl3, out, in_prod, wl1_prod, wl3_prod, out_cons
        ):
            tg = TaskGroup()
            in_prod.fill(inp, group=tg)
            wl1_prod.fill(wl1, group=tg)
            wl3_prod.fill(wl3, group=tg)
            out_cons.drain(out, wait=True, group=tg)
            tg.finish()

        rt = Runtime(
            sequence_with_wts,
            [
                in_ty,
                wts_ty,
                wts_ty,
                out_ty,
                act_in.prod(),
                wts_fifos[0].prod(),
                wts_fifos[1].prod(),
                out_fifo.cons(),
            ],
        )
    else:

        def sequence_no_wts(inp, out, in_prod, out_cons):
            tg = TaskGroup()
            in_prod.fill(inp, group=tg)
            out_cons.drain(out, wait=True, group=tg)
            tg.finish()

        rt = Runtime(
            sequence_no_wts,
            [
                in_ty,
                out_ty,
                act_in.prod(),
                out_fifo.cons(),
            ],
        )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=sa_placer_flags())
def per_block_design(
    *buffers: InOut,
    block_name: CompileTime[str],
    data_dir: CompileTime[str | None] = None,
    scales_json: CompileTime[str | None] = None,
    wts_tag: CompileTime[str] = "chain",
):
    return per_block_iron(block_name, data_dir, scales_json, wts_tag)


def _make_argparser():
    p = argparse.ArgumentParser(description="Build per-block IRON MLIR.")
    add_compile_args(p, default_dev="npu2")
    p.add_argument(
        "block", help="block name (bn0..bn3, bn4_5, bn6, bn7, bn8_9, bn10..bn14)"
    )
    p.add_argument(
        "--data-dir",
        help="weight files directory (default: ./data). Use this to point at "
        "bottleneck_A|B|C/data/ for per-bn fixture testing.",
    )
    p.add_argument(
        "--scales-json",
        help="scale_factors JSON file path (default: data/scale_factors_final.json). "
        "Use bottleneck_*/data/scale_factors.json for per-bn fixture testing.",
    )
    p.add_argument(
        "--sa-effort",
        type=float,
        help="SA placer search budget scale (default: 1.0; lower trades "
        "placement cost for compile time)",
    )
    return p


def main():
    opts = _make_argparser().parse_args()
    set_current_device(device_from_args(opts, n_cols=None))
    design = per_block_design
    if opts.sa_effort is not None:
        design = design.specialize(aiecc_flags=sa_placer_flags(effort=opts.sa_effort))
    compile_kwargs = dict(
        block_name=opts.block, data_dir=opts.data_dir, scales_json=opts.scales_json
    )
    if opts.xclbin_path:
        design.specialize(**compile_kwargs).compile(
            xclbin_path=opts.xclbin_path, inst_path=opts.insts_path
        )
    else:
        print(per_block_iron(**compile_kwargs))


if __name__ == "__main__":
    main()
