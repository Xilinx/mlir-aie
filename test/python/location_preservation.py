# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

import numpy as np

from aie.dialects.aie import (
    AIEDevice,
    AIETileType,
    DMAChannelDir,
    LockAction,
    WireBundle,
    buffer,
    core,
    device,
    dma_bd,
    end,
    flow,
    lock,
    mem,
    npu_write_rtp,
    packetflow,
    tile,
    use_lock,
)
from aie.dialects.aiex import (
    npu_address_patch,
    npu_maskwrite32,
    npu_push_queue,
    npu_rtp_write,
    npu_sync,
    npu_write32,
    runtime_sequence,
)
from aie.extras import types as T
from aie.extras.dialects.arith import constant
from aie.ir import Context, InsertionPoint, Location, Module
from aie.iron import Buffer, Flow, Lock, PacketDest, PacketFlow
from aie.iron.device import Tile


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
        # An unrelated ambient insertion point must not capture helper constants.
        other = Module.create()
        with InsertionPoint(other.body), Location.file("ambient.py", 1, 1):
            builder(ip=InsertionPoint(target), **kwargs)
        assert len(other.body.operations) == 0
    else:
        builder()
    emitted = list(target.operations)[start:]
    assert [op.operation.name for op in emitted] == names
    return emitted


def check_builder(builder, name, constants=0):
    names = ["arith.constant"] * constants + [name]
    for loc in (Location.file("explicit.py", 42, 7), Location.unknown()):
        for op in emitted_by(builder, names, loc=loc):
            assert_location(op.operation, loc)
    emitted = emitted_by(builder, names)
    loc = emitted[-1].operation.location
    # Check the actual user filename and line, not just a non-unknown location.
    assert f'"{__file__}":{builder.__code__.co_firstlineno}:' in str(loc), loc
    for op in emitted:
        assert_location(op.operation, loc)


def low_level_locations():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @device(AIEDevice.npu1)
            def device_body():
                src, dst = tile(0, 0), tile(0, 2)
                buf = buffer(dst, np.ndarray[(16,), np.dtype[np.int32]], name="rtp")
                lk = lock(dst, lock_id=0, init=1)
                dests = {"dest": dst, "port": WireBundle.DMA, "channel": 0}
                check_builder(
                    lambda **kw: flow(src, dest=dst, **kw),
                    "aie.flow",
                )
                check_builder(
                    lambda **kw: packetflow(0, src, WireBundle.DMA, 0, dests, **kw),
                    "aie.packetflow",
                )

                @core(dst)
                def core_body():
                    check_builder(
                        lambda **kw: use_lock(lk, LockAction.Acquire, **kw),
                        "aie.use_lock",
                        1,
                    )
                    value = constant(1, T.i32())
                    original_loc = value.owner.location
                    check_builder(
                        lambda **kw: use_lock(lk, LockAction.Release, value, **kw),
                        "aie.use_lock",
                    )
                    assert value.owner.location == original_loc

                @mem(dst)
                def mem_body():
                    loc = Location.file("dma.py", 12, 3)
                    for op in emitted_by(
                        lambda **kw: dma_bd(buf, offset=0, transfer_len=16, **kw),
                        ["aie.dma_bd"],
                        loc=loc,
                    ):
                        assert_location(op.operation, loc)
                    end()

                @runtime_sequence(np.ndarray[(16,), np.dtype[np.int32]])
                def sequence(_):
                    check_builder(
                        lambda **kw: npu_write32(np.int32(0x100), 7, **kw),
                        "aiex.npu.write32",
                        2,
                    )
                    check_builder(
                        lambda **kw: npu_maskwrite32(0x100, 7, 0xFF, **kw),
                        "aiex.npu.maskwrite32",
                        3,
                    )
                    check_builder(
                        lambda **kw: npu_sync(0, 0, 0, 0, **kw),
                        "aiex.npu.sync",
                        6,
                    )
                    check_builder(
                        lambda **kw: npu_address_patch(0x100, 0, 4, **kw),
                        "aiex.npu.address_patch",
                        1,
                    )
                    check_builder(
                        lambda **kw: npu_rtp_write("rtp", 0, 7, **kw),
                        "aiex.npu.rtp_write",
                        1,
                    )
                    check_builder(
                        lambda **kw: npu_write_rtp("rtp", 0, 7, **kw),
                        "aiex.npu.rtp_write",
                        1,
                    )
                    direction = DMAChannelDir.MM2S
                    check_builder(
                        lambda **kw: npu_push_queue(
                            0, 0, direction, 0, False, 0, 1, **kw
                        ),
                        "aiex.npu.push_queue",
                        2,
                    )
                    value = constant(7, T.i32())
                    original_loc = value.owner.location
                    check_builder(
                        lambda **kw: npu_write32(value, value, **kw),
                        "aiex.npu.write32",
                    )
                    assert value.owner.location == original_loc

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
                        ["aie.flow", "aie.shim_dma_allocation"],
                    ),
                    (
                        PacketFlow(
                            1,
                            dst,
                            src,
                            extra_dsts=[PacketDest(extra)],
                            shim_symbol="output",
                        ),
                        ["aie.packetflow", "aie.shim_dma_allocation"],
                    ),
                    (Lock(dst, lock_id=0, name="lock"), ["aie.lock"]),
                    (
                        Buffer(
                            np.ndarray[(16,), np.dtype[np.int32]], tile=dst, name="buf"
                        ),
                        ["aie.buffer"],
                    ),
                ]
                loc = Location.file("resolve.py", 27, 4)
                for obj, names in objects:
                    for op in emitted_by(obj.resolve, names, loc=loc):
                        assert_location(op.operation, loc)
                        if op.operation.name == "aie.packetflow":
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
