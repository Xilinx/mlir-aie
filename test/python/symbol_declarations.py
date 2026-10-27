# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.dialects import aievec
from aie.dialects._ods_common import get_op_result_or_value
from aie.dialects.aie import (
    AIEDevice,
    Core,
    DMAChannelDir,
    Device,
    LockAction,
    buffer,
    bds,
    dma_bd,
    end,
    external_buffer,
    external_func,
    find_matching_buffers,
    find_matching_locks,
    lock,
    object_fifo,
    tile,
    use_lock,
)
from aie.dialects.aiex import dma_configure_task, runtime_sequence
from aie.extras import types as T
from aie.extras.dialects.arith import constant
from aie.helpers.dialects.func import call
from aie.ir import Block, FlatSymbolRefAttr, InsertionPoint, StringAttr, SymbolTable
from util import construct_and_print_module


# CHECK-LABEL: symbolDeclarations
# CHECK: aie.buffer @buffer(
# CHECK: aie.external_buffer @buffer_0
# CHECK: aie.buffer @buffer_1(
# CHECK: aie.lock @lock(
# CHECK: aie.lock @lock_0(
# CHECK: aie.objectfifo.register_external_buffers @external_fifo (%{{.*}}, [@buffer_0])
# CHECK: aie.core(
# CHECK: aie.get_buffer @buffer_1 : memref<16xi32>
# CHECK: memref.load
# CHECK: aie.get_buffer @buffer_1 : memref<16xi32>
# CHECK: memref.store
# CHECK: aie.get_buffer @buffer_1 : memref<16xi32>
# CHECK: func.call @consume
# CHECK: aie.get_buffer @buffer_1 : memref<16xi32>
# CHECK: func.call @consume
# CHECK: aie.get_buffer @buffer_1 : memref<16xi32>
# CHECK: aievec.upd
# CHECK: aie.use_lock(@lock_0, Acquire,
# CHECK: aie.use_lock(@lock_0, Release,
# CHECK: aie.core(
# CHECK: aie.get_buffer @buffer_1 : memref<16xi32>
# CHECK: memref.load
@construct_and_print_module
def symbolDeclarations():
    dev = Device(AIEDevice.npu1_1col)
    with InsertionPoint(Block.create_at_start(dev.body_region)):
        t = tile(0, 2)
        t2 = tile(0, 3)
        shim = tile(0, 0)
        first = buffer(t, T.memref(16, T.i32()), name="buffer")
        ext = external_buffer(T.memref(16, T.i32()), name="buffer_0")
        b = buffer(t, T.memref(16, T.i32()))
        assert b.get_name() == "buffer_1"
        assert b.shape == (16,) and b.dtype == T.i32()
        assert b[:] is b and b[...] is b
        assert len(b.declaration.results) == len(ext.declaration.results) == 0
        assert b.owner == b.declaration.operation
        b.operation.attributes["metadata"] = StringAttr.get("retained")
        assert b.declaration.attributes["metadata"].value == "retained"
        assert b.sym_name.value == "buffer_1"
        named_lock = lock(t, sym_name="lock", lock_id=0)
        unnamed_lock = lock(t)
        assert unnamed_lock.sym_name.value == "lock_0"
        assert len(unnamed_lock.results) == 0
        assert find_matching_locks([t], sym_name="lock_0") == [unnamed_lock]
        assert [x.get_name() for x in find_matching_buffers([t])] == [
            "buffer",
            "buffer_1",
        ]
        consume = external_func("consume", [T.memref(16, T.i32())])
        fifo = object_fifo("external_fifo", shim, t, 1, T.memref(16, T.i32()))
        registration = fifo.register_external_buffers(shim, [ext])
        assert len(registration.operands) == 1
        assert isinstance(
            registration.attributes["externalBuffers"][0], FlatSymbolRefAttr
        )
        assert registration.attributes["externalBuffers"][0].value == "buffer_0"
        assert not any(
            op.operation.name == "aie.get_buffer"
            for op in dev.body_region.blocks[0].operations
        )
        for i in range(2):
            c = Core(t if i == 0 else t2)
            with InsertionPoint(Block.create_at_start(c.body)):
                b[0] = b[0]
                if i == 0:
                    consume(b)
                    call([], "consume", [b])
                    aievec.upd(T.vector(16, T.i32()), b, [constant(0, index=True)])
                    use_lock(unnamed_lock, LockAction.Acquire)
                    use_lock(unnamed_lock.sym_name, LockAction.Release, 1)
                    assert get_op_result_or_value(ext).owner.name == "aie.get_buffer"
                    nested = buffer(t, T.memref(4, T.i32()))
                    assert nested.declaration.operation.parent == dev.operation
                    assert len(nested.declaration.results) == 0
                end()
        assert first.declaration.sym_name.value == "buffer"
        assert named_lock.sym_name.value == "lock"


# CHECK-LABEL: symbolTableRename
# CHECK: aie.buffer @renamed_data(
# CHECK: aie.lock @renamed_gate(
# CHECK: aie.core(
# CHECK: aie.get_buffer @renamed_data : memref<16xi32>
# CHECK: aie.use_lock(@renamed_gate, Acquire,
# CHECK-NOT: @data
# CHECK-NOT: @gate
@construct_and_print_module
def symbolTableRename():
    dev = Device(AIEDevice.npu1_1col)
    with InsertionPoint(Block.create_at_start(dev.body_region)):
        t = tile(0, 2)
        b = buffer(t, T.memref(16, T.i32()), name="data")
        gate = lock(t, sym_name="gate")
        symbols = SymbolTable(dev.operation)
        data_decl = symbols["data"].operation
        gate_decl = symbols["gate"].operation
        assert data_decl == b.declaration.operation
        assert gate_decl == gate.operation
        assert len(data_decl.results) == len(gate_decl.results) == 0
        assert b.operation.name == "aie.buffer"
        assert b.operation.attributes["buffer_type"].value == b.type
        assert b.operation.parent == dev.operation
        assert "aie.buffer @data" in b.operation.get_asm()
        assert b.operation.verify()
        c = Core(t)
        with InsertionPoint(Block.create_at_start(c.body)):
            access = get_op_result_or_value(b)
            acquire = use_lock(gate, LockAction.Acquire)
            end()
        SymbolTable.replace_all_symbol_uses("data", "renamed_data", dev.operation)
        SymbolTable.set_symbol_name(data_decl, "renamed_data")
        SymbolTable.replace_all_symbol_uses("gate", "renamed_gate", dev.operation)
        SymbolTable.set_symbol_name(gate_decl, "renamed_gate")
        assert access.owner.attributes["buffer"].value == "renamed_data"
        assert acquire.attributes["lock"].value == "renamed_gate"
        assert b.get_name() == "renamed_data"
        assert gate.sym_name.value == "renamed_gate"
        renamed = SymbolTable(dev.operation)
        assert "data" not in renamed and "gate" not in renamed
        assert renamed["renamed_data"].operation == data_decl
        assert renamed["renamed_gate"].operation == gate_decl
        with InsertionPoint(c.body.blocks[0].operations[-1]):
            refreshed = get_op_result_or_value(b)
            assert refreshed.owner.attributes["buffer"].value == "renamed_data"


# CHECK-LABEL: explicitInsertionPoints
# CHECK: aie.core(
# CHECK: aie.get_buffer @local
# CHECK: func.call @consume
# CHECK: aie.get_buffer @local
# CHECK: func.call @consume
# CHECK: aie.get_buffer @local
# CHECK: func.call @consume
# CHECK: aie.get_buffer @local
# CHECK: aievec.upd
# CHECK: aie.use_lock(@gate, Acquire,
# CHECK: aie.core(
# CHECK-NEXT: aie.end
@construct_and_print_module
def explicitInsertionPoints():
    dev = Device(AIEDevice.npu1_1col)
    with InsertionPoint(Block.create_at_start(dev.body_region)):
        t = tile(0, 2)
        other = tile(0, 3)
        b = buffer(t, T.memref(16, T.i32()), name="local")
        gate = lock(t, sym_name="gate")
        consume = external_func("consume", [T.memref(16, T.i32())])
        target = Core(t)
        target_block = Block.create_at_start(target.body)
        with InsertionPoint(target_block):
            target_end = end()
        ip = InsertionPoint(target_end)
        ambient = Core(other)
        ambient_block = Block.create_at_start(ambient.body)
        with InsertionPoint(ambient_block):
            calls = [
                call(consume, [b], ip=ip),
                call([], "consume", [b], ip=ip),
                consume(b, ip=ip),
            ]
            for c in calls:
                assert c.operands[0].owner.operation.block == target_block
            index = constant(0, index=True, ip=ip)
            update = aievec.upd(T.vector(16, T.i32()), b, [index], ip=ip)
            assert update.owner.source.owner.operation.block == target_block
            acquire = use_lock(gate, LockAction.Acquire, ip=ip)
            assert acquire.operands[0].owner.operation.block == target_block
            end()
        assert len(ambient_block.operations) == 1


# CHECK-LABEL: explicitDmaInsertionPoint
# CHECK: aiex.dma_configure_task
# CHECK: %[[LOCAL:.*]] = aie.get_buffer @data
# CHECK-NEXT: aie.dma_bd(%[[LOCAL]]
# CHECK: aiex.dma_configure_task
# CHECK: aie.get_buffer @ambient_data
# CHECK-NOT: aie.get_buffer @data
@construct_and_print_module
def explicitDmaInsertionPoint():
    dev = Device(AIEDevice.npu1_1col)
    with InsertionPoint(Block.create_at_start(dev.body_region)):
        t = tile(0, 2)
        other = tile(0, 3)
        b = buffer(t, T.memref(16, T.i32()), name="data")
        ambient_buffer = buffer(other, T.memref(16, T.i32()), name="ambient_data")

        @runtime_sequence()
        def sequence():
            target = dma_configure_task(t, DMAChannelDir.MM2S, 0)
            with bds(target) as target_bds:
                with target_bds[0]:
                    target_end = end()
            ambient = dma_configure_task(other, DMAChannelDir.MM2S, 0)
            with bds(ambient) as ambient_bds:
                with ambient_bds[0]:
                    desc = dma_bd(b, ip=InsertionPoint(target_end))
                    assert desc.buffer.owner.operation.block == target_bds[0].block
                    dma_bd(ambient_buffer)
                    end()
