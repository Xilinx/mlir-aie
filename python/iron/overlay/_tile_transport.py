# _tile_transport.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Tile-sourced control-packet transport: a compute tile's own DMA writes a
ProgramMemorySlot, instead of the host.

Hardware-verified mechanism (test/npu-xrt/tile_sourced_ctrl_pkt_spike/), with
two non-obvious rules baked in as the only way this module builds a send:

1. A DMA channel gated by a lock nothing else ever touches (a
   "self-consuming, no real core action" trigger) never actually starts the
   hardware queue. Every chunk here is released by an actual core action.
2. Each control packet -- one address+opcode header plus up to 4 data words,
   the hardware "beats" field being 2 bits -- must be its own DMA burst.
   Combining chunks into one larger burst does not work. This is why a
   slot-sized payload becomes many chained `Bd` entries, not one.
"""

from dataclasses import dataclass

MAX_DATA_WORDS_PER_PACKET = 4


def control_packet_header(*, stream_id: int, opcode: int, size: int, addr: int) -> int:
    """The control-packet header word: address/opcode/beats + even parity.

    Matches `lib/Targets/AIETargetNPU.cpp`'s software-path formula exactly
    (validated byte-for-byte against `test/Targets/AIETargetCDODirect/
    control_packets.mlir`'s FileCheck'd output): the BD-native-tagged case
    (this module's only path) skips the *stream* header word the software
    path also builds -- that one carries routing information the hardware
    packetizer supplies out of band from the BD's own `packet=` tag instead.
    """
    beats = size - 1
    hdr = (stream_id & 0xFF) << 24 | (opcode & 0x3) << 22 | (beats & 0x3) << 20 | (
        addr & 0xFFFFF
    )

    def parity_even(n: int) -> bool:
        return bin(n).count("1") % 2 == 0

    return hdr | ((1 if parity_even(hdr) else 0) << 31)


@dataclass
class ControlPacketChunk:
    """One control packet: a header word, plus the ``size`` data words it writes."""

    header: int
    data: list[int]


def chunk_for_control_packets(host_addr: int, words: list[int]) -> list[ControlPacketChunk]:
    """Split `words` (to be written starting at `host_addr`) into control
    packets of at most `MAX_DATA_WORDS_PER_PACKET` data words each.

    Each chunk's header addresses exactly its own slice -- the address
    advances by `4 * len(chunk.data)` bytes per chunk, since program memory
    is word-addressed. Opcode 0 (write) throughout; ProgramMemorySlot's
    tile-sourced transport never reads.
    """
    chunks = []
    offset = 0
    for i in range(0, len(words), MAX_DATA_WORDS_PER_PACKET):
        data = words[i : i + MAX_DATA_WORDS_PER_PACKET]
        header = control_packet_header(
            stream_id=0, opcode=0, size=len(data), addr=host_addr + offset
        )
        chunks.append(ControlPacketChunk(header, data))
        offset += 4 * len(data)
    return chunks


def wire_words(chunks: list[ControlPacketChunk]) -> list[int]:
    """Flatten chunks into the single word array a source tile's payload
    global holds: [header0, data0..., header1, data1..., ...].
    """
    out = []
    for c in chunks:
        out.append(c.header)
        out.extend(c.data)
    return out


def done_chunk(addr: int) -> ControlPacketChunk:
    """The one-word "the write landed" signal chunk: a single `1` written to
    `addr` -- a plain data-memory flag on the target tile, not a hardware
    lock (writing a *lock's* value register needs its hardware-assigned
    address, which is not knowable until a much later compiler pass; a plain
    Buffer's address is knowable from the linked resident ELF, the same way
    `ProgramMemorySlot.pingpong()`'s bootstrap park's own address already is).
    Appended as its own trailing `Bd` after the main payload's iterated one,
    on the same `PacketFlow` (same destination tile, same TileControl port --
    routing is by destination and pkt_id, not by address).
    """
    return ControlPacketChunk(
        control_packet_header(stream_id=0, opcode=0, size=1, addr=addr), [1]
    )
