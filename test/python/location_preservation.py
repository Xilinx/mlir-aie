# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

import numpy as np

from aie.dialects.aie import (
    AIEDevice,
    AIETileType,
    WireBundle,
    device,
    external_func,
    flow,
    object_fifo,
    object_fifo_link,
    packetflow,
    tile,
)
from aie.ir import Context, InsertionPoint, Location, Module, loc_tracebacks
from aie.iron import Buffer, Flow, Lock, PacketDest, PacketFlow
from aie.iron.device import Tile

line_type = np.ndarray[(16,), np.dtype[np.int32]]


def assert_location(op, expected):
    assert op.location == expected, (op.name, op.location, expected)
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                assert_location(child.operation, expected)


def emitted_by(builder, names, **kwargs):
    target = InsertionPoint.current.block
    start = len(target.operations)
    if kwargs:
        # An unrelated ambient insertion point and location must not win.
        other = Module.create()
        with InsertionPoint(other.body), Location.file("ambient.py", 1, 1):
            builder(ip=InsertionPoint(target), **kwargs)
        assert len(other.body.operations) == 0
    else:
        builder()
    emitted = list(target.operations)[start:]
    assert [op.operation.name for op in emitted] == names, [
        op.operation.name for op in emitted
    ]
    return emitted


def check_builder(builder, name):
    for loc in (Location.file("explicit.py", 42, 7), Location.unknown()):
        for op in emitted_by(builder, [name], loc=loc):
            assert_location(op.operation, loc)
    with loc_tracebacks(max_depth=1):
        (op,) = emitted_by(builder, [name])
    loc = op.operation.location.child_loc
    assert loc.filename == __file__, loc
    assert loc.start_line == builder.__code__.co_firstlineno, loc


def low_level_locations():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @device(AIEDevice.npu1)
            def device_body():
                src, mem, dst = tile(0, 0), tile(0, 1), tile(0, 2)
                pkt_src = tile(3, 0)
                dests = {"dest": tile(3, 2), "port": WireBundle.DMA, "channel": 0}
                destinations = iter([dst, tile(1, 2), tile(2, 2)])
                check_builder(
                    lambda **kw: flow(src, dest=next(destinations), **kw),
                    "aie.flow",
                )
                packet_ids = iter(range(3))
                check_builder(
                    lambda **kw: packetflow(
                        next(packet_ids), pkt_src, WireBundle.DMA, 0, dests, **kw
                    ),
                    "aie.packet_flow",
                )
                fifo_names = iter(f"of{i}" for i in range(3))
                check_builder(
                    lambda **kw: object_fifo(
                        next(fifo_names), src, [mem], 2, line_type, **kw
                    ),
                    "aie.objectfifo",
                )
                links = iter(
                    [
                        (
                            object_fifo(f"in{i}", src, [mem], 2, line_type),
                            object_fifo(f"out{i}", mem, [dst], 2, line_type),
                        )
                        for i in range(3)
                    ]
                )
                check_builder(
                    lambda **kw: object_fifo_link(*next(links), **kw),
                    "aie.objectfifo.link",
                )
                func_names = iter(f"kernel{i}" for i in range(3))
                check_builder(
                    lambda **kw: external_func(next(func_names), [line_type], **kw),
                    "func.func",
                )

        assert module.operation.verify()


def iron_locations():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @device(AIEDevice.npu1)
            def device_body():
                src = Tile(0, 0, tile_type=AIETileType.ShimNOCTile)
                dst = Tile(0, 2, tile_type=AIETileType.CoreTile)
                extra = Tile(0, 3, tile_type=AIETileType.CoreTile)
                for t in (src, dst, extra):
                    t.op = tile(t.col, t.row)
                objects = [
                    (
                        Flow(src, dst, shim_symbol="input"),
                        ["aie.route_endpoint", "aie.route_endpoint", "aie.route"],
                    ),
                    (
                        PacketFlow(
                            1,
                            src,
                            dst,
                            src_channel=1,
                            dst_channel=1,
                            extra_dsts=[PacketDest(extra)],
                            shim_symbol="output",
                        ),
                        ["aie.packet_flow", "aie.shim_dma_allocation"],
                    ),
                    (Lock(dst, lock_id=0, name="lock"), ["aie.lock"]),
                    (Buffer(line_type, tile=dst, name="buf"), ["aie.buffer"]),
                ]
                loc = Location.file("resolve.py", 27, 4)
                for obj, names in objects:
                    for op in emitted_by(obj.resolve, names, loc=loc):
                        assert_location(op.operation, loc)
                        if op.operation.name == "aie.packet_flow":
                            assert [
                                child.operation.name
                                for child in op.regions[0].blocks[0].operations
                            ] == [
                                "aie.packet_source",
                                "aie.packet_dest",
                                "aie.packet_dest",
                                "aie.end",
                            ]
                    emitted_by(obj.resolve, [], loc=Location.file("second.py", 1, 1))

        assert module.operation.verify()


low_level_locations()
iron_locations()
