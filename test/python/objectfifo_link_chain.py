# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

"""Fifos that chain two ObjectFifoLinks: one link's output and the next's
input. aie.iron resolves the links' tiles through them, the placer keeps
each pair of chained links on two memtiles, and two links on one tile are
refused."""

import numpy as np
import pytest
from aie import ir
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.dialects._aie_enum_gen import AIETileType
from aie.passmanager import PassManager

PLACE = (
    "builtin.module(aie.device(aie-place-tiles),"
    "aie.device(aie-objectFifo-stateful-transform))"
)


def vector(n):
    return np.ndarray[(n,), np.dtype[np.int32]]


def gather(fan_ins: list[int], tile=None) -> str:
    """Cores' 16-element objects joined into one stream drained to the host:
    the stream a join of ``fan_ins[0]`` fifos, each of those a join of
    ``fan_ins[1]``, and so on, every link on ``tile`` if given.
    """
    lanes = int(np.prod(fan_ins))
    out = ObjectFifo(vector(16 * lanes), depth=2)
    level = [out]
    for parts in fan_ins:
        width = 16 * lanes // len(level) // parts
        level = [
            sub
            for fifo in level
            for sub in fifo.prod().join(
                [k * width for k in range(parts)],
                **({} if tile is None else {"tile": tile}),
                obj_types=[vector(width)] * parts,
            )
        ]

    def produce(of):
        of.acquire(1)
        of.release(1)

    workers = [Worker(produce, [lane.prod()]) for lane in level]

    def sequence(C, drained):
        drained.drain(C, wait=True)

    rt = Runtime(sequence, [vector(16 * lanes), out.cons()])
    return str(Program(NPU2(), rt, workers=workers).resolve_program())


def place(text: str) -> ir.Module:
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.parse(text)
        PassManager.parse(PLACE, ctx).run(module.operation)
        return module


def test_join_of_joins_places_each_link_on_its_own_memtile():
    text = gather([2, 4])
    assert text.count("aie.objectfifo.link") == 3
    placed = str(place(text))
    assert placed.count("aie.memtile_dma(") == 3


def test_a_chain_of_three_links_resolves():
    """The middle joins' tiles are reached only through the fifos between
    links, which no worker or runtime holds an end of."""
    text = gather([2, 2, 2])
    assert text.count("aie.objectfifo.link") == 7
    place(text)


def test_two_chained_links_on_one_tile_are_refused():
    shared = Tile(1, 1, tile_type=AIETileType.MemTile)
    with pytest.raises(ir.MLIRError, match="chains two ObjectFifoLinkOps on one tile"):
        gather([2, 2], tile=shared)
