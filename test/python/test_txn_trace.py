# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

"""Unit tests for aie.utils.txn_trace: no NPU required.

The shim streams are assembled by hand with the word layouts from
include/aie/Runtime/TxnEncoding.h, mirroring what the static emitter and the
dynamic (dispatch-time) builder produce for the same transfers. The memtile
and core streams are lowered from runtime-sequence DMA tasks by the compiler.
"""

import numpy as np
import pytest
from aie.dialects.aie import translate_npu_to_binary
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager
from aie.utils.txn_trace import compare, decode, explain, trace

SHIM_BD0 = 0x1D000
S2MM0_CTRL, S2MM0_QUEUE = 0x1D200, 0x1D204
MM2S0_CTRL, MM2S0_QUEUE = 0x1D210, 0x1D214
TOKEN = 1 << 31


# Each helper returns a list of ops, so a stream is a sum of helper calls.


def header(n_ops, n_words, dev_gen=4, rows=6, cols=4, mem_tile_rows=1):
    return [(rows << 24) | (dev_gen << 16) | (1 << 8), (mem_tile_rows << 8) | cols] + [
        n_ops,
        4 * n_words,
    ]


def write32(addr, val):
    return [[0, 0, addr, 0, val, 24]]


def maskwrite32(addr, val, mask):
    return [[3, 0, addr, 0, val, mask, 28]]


def maskpoll32(addr, val, mask):
    return [[4, 0, addr, 0, val, mask, 28]]


def blockwrite(addr, data, col=0, row=0):
    return [[1, col | (row << 8), addr, 4 * (4 + len(data))] + list(data)]


def patch(addr, arg_idx, arg_plus):
    return [
        [
            129,
            48,
            0,
            0,
            0,
            0,
            addr,
            0,
            arg_idx,
            0,
            arg_plus & 0xFFFFFFFF,
            arg_plus >> 32,
        ]
    ]


def tct(col, row, direction, channel, ncol=1, nrow=1):
    return [
        [
            128,
            16,
            direction | (row << 8) | (col << 16),
            (nrow << 8) | (ncol << 16) | (channel << 24),
        ]
    ]


def bd_words(
    length,
    *,
    offset=0,
    d0=None,
    d1=None,
    d2_stride=1,
    iteration=None,
    valid=True,
    next_bd=None,
):
    """Shim BD words, static-emitter style: linear when no dims are given."""
    w = [length, offset, 0, 0, 0, 0, 0, 0]
    if d0 is not None:
        w[3] = ((d0[0] & 0x3FF) << 20) | ((d0[1] - 1) & 0xFFFFF)
    if d1 is not None:
        w[4] = ((d1[0] & 0x3FF) << 20) | ((d1[1] - 1) & 0xFFFFF)
    w[4] |= 2 << 30  # burst length encoding
    w[5] = (d2_stride - 1) & 0xFFFFF if (d0 or d1) else 0
    w[5] |= 2 << 24  # AXCache
    if iteration is not None:
        w[6] = ((iteration[0] & 0x3F) << 20) | ((iteration[1] - 1) & 0xFFFFF)
    w[7] = (1 << 25) if valid else 0
    if next_bd is not None:
        w[7] |= (1 << 26) | (next_bd << 27)
    return w


def transfer_static(
    bd_id, arg_idx, byte_off, length, queue, ctrl=None, token=False, **dims
):
    """Build a static-path transfer.

    BD image with the offset folded into word 1, an address patch carrying it
    again, then the queue push.
    """
    bd = SHIM_BD0 + 0x20 * bd_id
    s = blockwrite(bd, bd_words(length, offset=byte_off, **dims))
    s += patch(bd + 4, arg_idx, byte_off)
    if ctrl is not None:
        s += maskwrite32(queue - 4, ctrl, 0x1F00)
    s += write32(queue, bd_id | (TOKEN if token else 0))
    return s


def transfer_dynamic(
    bd_id, arg_idx, byte_off, length, queue, ctrl=None, token=False, unit_wraps=True
):
    """Build the same transfer as the dispatch-time builder emits it.

    A BD from the pool (any id), word 1 left zero, unit wraps instead of
    linear mode, a poll for room in the task queue.
    """
    bd = SHIM_BD0 + 0x20 * bd_id
    dims = dict(d0=(1, 1), d1=(1, 1), d2_stride=1) if unit_wraps else {}
    s = maskpoll32(0x1D228, 0, 1 << 22)
    s += blockwrite(bd, bd_words(length, **dims))
    s += patch(bd + 4, arg_idx, byte_off)
    if ctrl is not None:
        s += maskwrite32(queue - 4, ctrl, 0x1F00)
    s += write32(queue, bd_id | (TOKEN if token else 0))
    return s


def as_stream(*parts, **device):
    ops = [op for part in parts for op in part]
    words = [w for op in ops for w in op]
    return np.array(header(len(ops), 4 + len(words), **device) + words, dtype=np.uint32)


class TestDecode:
    def test_round_trip_of_every_opcode(self):
        s = as_stream(
            write32(0x1D214, 1),
            maskwrite32(0x1D200, 0xF00, 0x1F00),
            maskpoll32(0x1D228, 0, 0x400000),
            blockwrite(SHIM_BD0, bd_words(64)),
            patch(SHIM_BD0 + 4, 1, 0x400),
            tct(0, 0, 0, 0),
            [[8 | (3 << 16), 0x100, 0x1000, 0]],  # loadpdi id 3
            [[6 | (2 << 8)]],  # preempt level 2
            [[10 | (1 << 8), 0x400, 0, 0]],  # create_scratchpad, usage 1
            [[12 | (2 << 8) | (1 << 16), 0x10, 0x1D004]],  # update_reg
        )
        kinds = [op.kind for op in decode(s)]
        assert kinds == [
            "header",
            "write32",
            "maskwrite32",
            "maskpoll32",
            "blockwrite",
            "patch",
            "tct",
            "loadpdi",
            "preempt",
            "scratchpad",
            "update_reg",
        ]
        # Every op prints, and the positions chain without gaps.
        ops = decode(s)
        assert all(str(op) for op in ops)
        assert ops[-1].pos + len(ops[-1].words) == len(s)

    def test_rejects_unknown_opcode_and_short_header(self):
        with pytest.raises(ValueError, match="unknown TXN opcode"):
            decode(as_stream([[0x42, 0, 0, 0]]))
        with pytest.raises(ValueError, match="header"):
            decode([0, 0])

    def test_header_must_match_the_stream(self):
        s = as_stream(write32(MM2S0_QUEUE, 0), write32(MM2S0_QUEUE, 0))
        with pytest.raises(ValueError, match="gives 64 bytes, the stream has 40"):
            decode(s[:10])
        s[2] = 3
        with pytest.raises(ValueError, match="counts 3 ops, the stream has 2"):
            decode(s)


class TestTrace:
    def test_push_resolves_bd_patch_and_control(self):
        s = as_stream(
            transfer_static(0, 1, 0x400, 256, S2MM0_QUEUE, ctrl=0xF00, token=True),
            transfer_static(1, 0, 0x400, 256, MM2S0_QUEUE),
            tct(0, 0, 0, 0),
        )
        ev = trace(s)
        assert [e.kind for e in ev] == ["push", "push", "wait"]
        first = ev[0]
        assert (first.col, first.row, first.direction, first.channel) == (
            0,
            0,
            "S2MM",
            0,
        )
        assert first.issue_token and first.ctrl == 0xF00
        assert first.bd.length == 256
        assert first.bd.address == ("arg", 1, 0x400)
        assert first.bd.dims == () and first.bd.outer_stride == 1
        assert not ev[1].issue_token and ev[1].bd.address == ("arg", 0, 0x400)
        assert ev[2].direction == "S2MM" and ev[2].channel == 0

    def test_unpatched_bd_keeps_absolute_address(self):
        bd = SHIM_BD0
        s = as_stream(
            blockwrite(bd, bd_words(16, offset=0x1234)), write32(MM2S0_QUEUE, 0)
        )
        (ev,) = trace(s)
        assert ev.bd.address == ("abs", 0x1234)

    def test_nd_dims_and_repeat(self):
        w = bd_words(1024, d0=(32, 1), d1=(8, 64), d2_stride=2048, iteration=(4, 512))
        s = as_stream(blockwrite(SHIM_BD0, w), write32(MM2S0_QUEUE, 0 | (3 << 16)))
        (ev,) = trace(s)
        assert ev.repeat == 3
        assert ev.bd.dims == ((32, 1), (8, 64))
        assert ev.bd.outer_stride == 2048
        assert ev.bd.iteration == (4, 512)

    def test_column_and_row_come_from_the_address(self):
        col, row = 2, 0
        tile = (col << 25) | (row << 20)
        s = as_stream(
            blockwrite(tile | SHIM_BD0, bd_words(8)), write32(tile | S2MM0_QUEUE, 0)
        )
        (ev,) = trace(s)
        assert (ev.col, ev.row) == (col, row)

    def test_unwritten_bd_is_an_error_not_zeros(self):
        with pytest.raises(ValueError, match=r"never writes its words \[0, 1, 2"):
            trace(as_stream(write32(MM2S0_QUEUE, 3)))
        partial = as_stream(write32(SHIM_BD0, 64), write32(MM2S0_QUEUE, 0))
        with pytest.raises(ValueError, match=r"BD 0 of tile \(0,0\).*\[1, 2, 3"):
            trace(partial)

    def test_push_follows_the_bd_chain(self):
        s = as_stream(
            blockwrite(SHIM_BD0, bd_words(64, next_bd=5)),
            blockwrite(SHIM_BD0 + 5 * 0x20, bd_words(32, offset=0x100, next_bd=2)),
            blockwrite(SHIM_BD0 + 2 * 0x20, bd_words(16, offset=0x200)),
            write32(MM2S0_QUEUE, 0),
        )
        (ev,) = trace(s)
        assert ev.bd.length == 64
        assert [t.length for t in ev.chain] == [32, 16]
        assert [t.address for t in ev.chain] == [("abs", 0x100), ("abs", 0x200)]
        assert ev.chain_loop is None
        assert "-> len=32" in explain(s)

    def test_cyclic_chain_records_where_it_loops(self):
        s = as_stream(
            blockwrite(SHIM_BD0, bd_words(64, next_bd=1)),
            blockwrite(SHIM_BD0 + 0x20, bd_words(32, next_bd=0)),
            write32(MM2S0_QUEUE, 0),
        )
        (ev,) = trace(s)
        assert [t.length for t in ev.chain] == [32] and ev.chain_loop == 0

    def test_register_writes_outside_the_dma_are_events(self):
        rtp = (2 << 20) | 0x400  # core (0, 2) data memory
        s = as_stream(
            write32(rtp, 7),
            maskwrite32((2 << 20) | 0x1F000, 1, 0x3F),
            blockwrite(rtp + 8, [5, 6]),
            transfer_static(0, 1, 0, 256, S2MM0_QUEUE, ctrl=0xF00),
        )
        writes = [e for e in trace(s) if e.kind == "write"]
        assert [(e.col, e.row, e.raw) for e in writes] == [
            (0, 2, (0x400, 7, 0xFFFFFFFF)),
            (0, 2, (0x1F000, 1, 0x3F)),
            (0, 2, (0x408, 5, 0xFFFFFFFF)),
            (0, 2, (0x40C, 6, 0xFFFFFFFF)),
        ]
        assert "write (0,2) 0x1f000 <- 0x00000001 & 0x0000003f" in explain(s)

    def test_memtile_and_core_rows_come_from_the_header(self):
        # With two memtile rows, row 2 is a memtile: its BD 37 is in range.
        tile = 2 << 20
        s = as_stream(
            blockwrite(tile | (0xA0000 + 37 * 0x20), [64, 0, 0, 0, 0, 0, 0, 1 << 31]),
            write32(tile | 0xA0604, 37),
            mem_tile_rows=2,
        )
        (ev,) = trace(s)
        assert (ev.row, ev.bd.length) == (2, 64)


class TestCompare:
    def test_different_device_is_reported(self):
        transfer = transfer_static(0, 1, 0, 256, S2MM0_QUEUE)
        (msg,) = compare(as_stream(transfer, dev_gen=3), as_stream(transfer))
        assert msg.startswith("headers differ")

    def test_different_runtime_parameter_is_reported(self):
        rtp = (2 << 20) | 0x400
        transfer = transfer_static(0, 1, 0, 256, S2MM0_QUEUE)
        a = as_stream(write32(rtp, 128), transfer)
        b = as_stream(write32(rtp, 64), transfer)
        (msg,) = compare(a, b)
        assert "event 0 differs" in msg and "0x00000080" in msg

    def test_chained_bd_difference_is_reported(self):
        def chained(bd_a, bd_b, tail_len):
            return as_stream(
                blockwrite(SHIM_BD0 + bd_a * 0x20, bd_words(64, next_bd=bd_b)),
                blockwrite(SHIM_BD0 + bd_b * 0x20, bd_words(tail_len)),
                write32(MM2S0_QUEUE, bd_a),
            )

        assert compare(chained(0, 1, 32), chained(6, 3, 32)) == []
        assert compare(chained(0, 1, 32), chained(0, 1, 16))

    def test_dynamic_and_static_encodings_of_one_transfer_are_equivalent(self):
        static = as_stream(
            transfer_static(0, 1, 0x800, 256, S2MM0_QUEUE, ctrl=0xF00, token=True),
            transfer_static(1, 0, 0x800, 256, MM2S0_QUEUE),
            tct(0, 0, 0, 0),
        )
        dynamic = as_stream(
            transfer_dynamic(3, 1, 0x800, 256, S2MM0_QUEUE, ctrl=0xF00, token=True),
            transfer_dynamic(5, 0, 0x800, 256, MM2S0_QUEUE),
            tct(0, 0, 0, 0),
        )
        assert len(static) != len(dynamic)  # not byte-identical...
        assert compare(static, dynamic) == []  # ...but the same DMA events

    def test_different_offset_is_reported(self):
        a = as_stream(transfer_static(0, 1, 0x800, 256, S2MM0_QUEUE))
        b = as_stream(transfer_dynamic(0, 1, 0xC00, 256, S2MM0_QUEUE))
        (msg,) = compare(a, b, names=("static", "dynamic"))
        assert "event 0 differs" in msg and "arg1+0x800" in msg and "arg1+0xc00" in msg

    def test_missing_wait_is_reported(self):
        a = as_stream(
            transfer_static(0, 1, 0, 256, S2MM0_QUEUE, token=True), tct(0, 0, 0, 0)
        )
        b = as_stream(transfer_static(0, 1, 0, 256, S2MM0_QUEUE, token=True))
        assert compare(a, b) == ["a has 2 events, b has 1"]

    def test_real_dimension_difference_is_not_normalized_away(self):
        a = as_stream(
            blockwrite(SHIM_BD0, bd_words(64, d0=(8, 1), d1=(8, 16))),
            write32(MM2S0_QUEUE, 0),
        )
        b = as_stream(
            blockwrite(SHIM_BD0, bd_words(64, d0=(8, 1), d1=(8, 8))),
            write32(MM2S0_QUEUE, 0),
        )
        assert compare(a, b)

    def test_unit_wraps_with_a_non_unit_outer_stride_differ_from_linear(self):
        linear = as_stream(blockwrite(SHIM_BD0, bd_words(64)), write32(MM2S0_QUEUE, 0))
        strided = as_stream(
            blockwrite(SHIM_BD0, bd_words(64, d0=(1, 1), d1=(1, 1), d2_stride=2)),
            write32(MM2S0_QUEUE, 0),
        )
        assert compare(linear, strided)

    def test_contiguous_nd_equals_linear(self):
        # [32 x 1][128 x 32] with the next block at +4096 is a linear 4096-word
        # scan; the static emitter folds it, the dynamic builder cannot when a
        # stride is a runtime value.
        linear = as_stream(
            blockwrite(SHIM_BD0, bd_words(4096)), write32(S2MM0_QUEUE, 0)
        )
        nd = as_stream(
            blockwrite(
                SHIM_BD0, bd_words(4096, d0=(32, 1), d1=(128, 32), d2_stride=4096)
            ),
            write32(S2MM0_QUEUE, 0),
        )
        assert compare(linear, nd) == []
        # A second pass over the dims needs the next block to follow on: a gap
        # between blocks is a real difference, and so is a non-unit stride.
        two_blocks = as_stream(
            blockwrite(
                SHIM_BD0, bd_words(8192, d0=(32, 1), d1=(128, 32), d2_stride=4096)
            ),
            write32(S2MM0_QUEUE, 0),
        )
        linear2 = as_stream(
            blockwrite(SHIM_BD0, bd_words(8192)), write32(S2MM0_QUEUE, 0)
        )
        assert compare(linear2, two_blocks) == []
        gapped = as_stream(
            blockwrite(
                SHIM_BD0, bd_words(8192, d0=(32, 1), d1=(128, 32), d2_stride=8192)
            ),
            write32(S2MM0_QUEUE, 0),
        )
        assert compare(two_blocks, gapped)
        strided = as_stream(
            blockwrite(SHIM_BD0, bd_words(4096, d0=(32, 1), d1=(128, 64))),
            write32(S2MM0_QUEUE, 0),
        )
        assert compare(linear, strided)

    def test_contiguous_repeat_folds_into_length(self):
        # Two executions (repeat 1) of a 4096-word linear BD whose iteration
        # dimension steps by 4096 is one 8192-word transfer.
        folded = as_stream(
            blockwrite(SHIM_BD0, bd_words(8192)), write32(S2MM0_QUEUE, 0)
        )
        repeated = as_stream(
            blockwrite(SHIM_BD0, bd_words(4096, iteration=(1, 4096))),
            write32(S2MM0_QUEUE, 0 | (1 << 16)),
        )
        assert compare(folded, repeated) == []
        # A repeat that re-reads the same block (no iteration step) is not.
        rereading = as_stream(
            blockwrite(SHIM_BD0, bd_words(4096)), write32(S2MM0_QUEUE, 0 | (1 << 16))
        )
        assert compare(folded, rereading)

    def test_explain_lists_events(self):
        s = as_stream(
            transfer_static(0, 1, 0x400, 256, S2MM0_QUEUE, token=True), tct(0, 0, 0, 0)
        )
        text = explain(s)
        assert "push (0,0) S2MM ch0" in text and "wait (0,0) S2MM ch0" in text
        assert "blockwrite" in explain(s, raw=True)


def lowered(device, *tasks):
    """The instruction stream for runtime-sequence DMA tasks on tile (0, 1)
    (`%mem`, a 1024-word buffer `%mbuf`) and tile (0, 2) (`%core`, `%cbuf`)."""
    src = f"""
    module {{
      aie.device({device}) {{
        %shim = aie.tile(0, 0)
        %mem = aie.tile(0, 1)
        %core = aie.tile(0, 2)
        %mbuf = aie.buffer(%mem) {{address = 0x4000 : i32}} : memref<1024xi32>
        %cbuf = aie.buffer(%core) {{address = 0x800 : i32}} : memref<256xi32>
        aie.runtime_sequence() {{
          {"".join(tasks)}
        }}
      }}
    }}"""
    with Context(), Location.unknown():
        module = Module.parse(src)
        PassManager.parse(
            "builtin.module(aie.device(aie-dma-tasks-to-npu,aie-dma-to-npu))"
        ).run(module.operation)
        return translate_npu_to_binary(module.operation)


def task(tile, direction, channel, bd, bd_id, attrs=""):
    name = f"%t{bd_id}"
    return f"""
          {name} = aiex.dma_configure_task(%{tile}, {direction}, {channel}) {{
            aie.dma_bd({bd}) {{bd_id = {bd_id} : i32}}
            aie.end
          }} {attrs}
          aiex.dma_start_task({name})"""


PAD = (
    "pad [<const_pad_before=0, const_pad_after=0>, "
    "<const_pad_before=1, const_pad_after=2>, "
    "<const_pad_before=0, const_pad_after=0>]"
)
MEM_ND = "%mbuf : memref<1024xi32> offset = 8 len = 192 sizes = [4, 6, 8] strides = [64, 8, 1]"


@pytest.mark.parametrize("device", ["npu1", "npu2"])
class TestLoweredTiles:
    def test_memtile_bd_above_15(self, device):
        s = lowered(
            device,
            task(
                "mem",
                "MM2S",
                3,
                f"{MEM_ND} {PAD}",
                37,
                "{repeat_count = 2 : i32, issue_token = true}",
            ),
        )
        (ev,) = trace(s)
        assert (ev.col, ev.row, ev.direction, ev.channel) == (0, 1, "MM2S", 3)
        assert ev.repeat == 2 and ev.issue_token
        assert ev.bd.length == 192
        assert ev.bd.dims == ((8, 1), (6, 8)) and ev.bd.outer_stride == 64
        assert ev.bd.padding == ((0, 0), (1, 2), (0, 0))
        assert "pad=[0/0 1/2 0/0]" in explain(s)

    def test_memtile_bd_id_and_padding_compare(self, device):
        bd = f"{MEM_ND} {PAD}"
        low = lowered(device, task("mem", "MM2S", 0, bd, 2))
        high = lowered(device, task("mem", "MM2S", 0, bd, 41))
        assert compare(low, high) == []
        unpadded = lowered(device, task("mem", "MM2S", 0, MEM_ND, 41))
        assert compare(high, unpadded)

    def test_memtile_contiguous_nd_equals_linear(self, device):
        linear = lowered(
            device, task("mem", "S2MM", 1, "%mbuf : memref<1024xi32> len = 512", 20)
        )
        nd = lowered(
            device,
            task(
                "mem",
                "S2MM",
                1,
                "%mbuf : memref<1024xi32> len = 512 sizes = [2, 16, 16] "
                "strides = [256, 16, 1]",
                20,
            ),
        )
        assert compare(linear, nd) == []
        strided = lowered(
            device,
            task(
                "mem",
                "S2MM",
                1,
                "%mbuf : memref<1024xi32> len = 512 sizes = [2, 16, 16] "
                "strides = [256, 32, 1]",
                20,
            ),
        )
        assert compare(linear, strided)

    def test_memtile_bd_chain(self, device):
        def chain(second_len):
            return f"""
          %t = aiex.dma_configure_task(%mem, MM2S, 0) {{
            aie.dma_bd(%mbuf : memref<1024xi32> len = 64) {{bd_id = 4 : i32}}
            aie.next_bd ^bd1
          ^bd1:
            aie.dma_bd(%mbuf : memref<1024xi32> offset = 64 len = {second_len}) {{bd_id = 5 : i32}}
            aie.end
          }}
          aiex.dma_start_task(%t)"""

        (ev,) = trace(lowered(device, chain(64)))
        assert ev.bd.length == 64
        ((length, (_, address)),) = [(t.length, t.address) for t in ev.chain]
        assert (length, address) == (64, ev.bd.address[1] + 64)
        assert compare(lowered(device, chain(64)), lowered(device, chain(32)))

    def test_core_bd(self, device):
        s = lowered(
            device,
            task(
                "core",
                "S2MM",
                1,
                "%cbuf : memref<256xi32> offset = 4 len = 64 sizes = [8, 8] "
                "strides = [16, 1]",
                9,
            ),
        )
        (ev,) = trace(s)
        assert (ev.col, ev.row, ev.direction, ev.channel) == (0, 2, "S2MM", 1)
        assert ev.bd.length == 64
        assert ev.bd.address == ("abs", (0x800 + 4 * 4) // 4)
        assert ev.bd.dims == ((8, 1), (8, 16))
        assert ev.bd.padding == ()
