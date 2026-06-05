#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""MobileNet V3 — physical tile placement (algorithm/mapping split).

Compute tiles use rows 2-5 of columns 0-7 (Strix layout). MemTiles live on
row 1; shim DMA endpoints on row 0. Adjacent rows in the same column share
memory, which we exploit for fused-pair self-loop fifos and L2->L3 handoffs.

Re-targeting the design = editing this dict (and only this dict). The logical
network shape lives in network_spec.py and has no awareness of these tiles.
"""

from aie.iron.device import Tile

PLACEMENT = {
    "init": Tile(0, 2),
    "post_l1": {"compute": Tile(6, 4), "memtile": Tile(4, 1)},
    "post_l2": {
        "wts_memtiles": [Tile(1, 1), Tile(3, 1), Tile(5, 1), Tile(7, 1)],
        "compute": [Tile(6, 3), Tile(7, 4), Tile(7, 3), Tile(7, 2)],
        "join_memtile": Tile(6, 1),
    },
    "shim": {
        "input": Tile(0, 0),
        "wts": [Tile(c, 0) for c in (4, 5, 6, 7)],
        "scratch_drain": Tile(3, 0),
        "fc_fill": Tile(4, 0),
        "fc_drain": Tile(7, 0),
    },
    "regular": {
        "bn0": Tile(0, 3),
        "bn1": Tile(0, 4),
        "bn2": Tile(0, 5),
        "bn3": Tile(1, 3),
        "bn4_5": {"compute": Tile(1, 2), "alloc": Tile(0, 2)},  # alloc on init tile
        "bn6": Tile(1, 4),
        "bn7": Tile(2, 3),
        "bn8_9": {"compute": Tile(3, 3), "alloc": Tile(3, 4)},  # alloc on bn11 L1 tile
    },
    "pipeline": {
        "bn10": {"l1": Tile(1, 5), "l2": Tile(2, 4), "l3": Tile(2, 5)},
        "bn11": {
            "l1": Tile(3, 2),
            "l2": Tile(3, 4),
            "l3": Tile(2, 2),
            "mem_skip": Tile(2, 1),
        },
        "bn12": {"l1": Tile(3, 5), "l23": Tile(4, 4)},
    },
    "cascade": {
        "bn13": {
            "l1_put": Tile(4, 5),
            "l1_get": Tile(5, 5),
            "l2": Tile(5, 4),
            "l3_put": Tile(4, 3),
            "l3_get": Tile(5, 3),
            "mem_l1": Tile(0, 1),
            "mem_l3": Tile(1, 1),
            "mem_skip": Tile(5, 1),
        },
        "bn14": {
            "l1_put": Tile(6, 5),
            "l1_get": Tile(7, 5),
            "l2": Tile(6, 2),
            "l3_put": Tile(4, 2),
            "l3_get": Tile(5, 2),
            "mem_l1": Tile(2, 1),
            "mem_l3": Tile(3, 1),
            "mem_skip": Tile(7, 1),
        },
    },
}
