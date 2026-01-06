#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# RUN: %python %s | FileCheck %s
#
# This test verifies event enums are correctly generated for AIE1, AIE2, and AIE2P.
# CHECK: Testing AIE1
# CHECK: AIE1 CoreEvent: NONE=0, TRUE=1, PERF_CNT_0=5, BROADCAST_15=122
# CHECK: AIE1 MemEvent: NONE=0, DMA_S2MM_0_FINISHED_BD=25
# CHECK: AIE1 ShimEvent: NONE=0, DMA_S2MM_0_FINISHED_BD=16
# CHECK: Testing AIE2
# CHECK: AIE2 CoreEvent: NONE=0, TRUE=1, PERF_CNT_0=5, BROADCAST_0=107
# CHECK: AIE2 MemEvent: NONE=0, WATCHPOINT_0=16, DMA_S2MM_0_START_TASK=19
# CHECK: AIE2 ShimEvent: NONE=0, DMA_S2MM_0_START_TASK=14
# CHECK: AIE2 MemTileEvent: NONE=0, CONFLICT_DM_BANK_0=112
# CHECK: Testing AIE2P
# CHECK: AIE2P CoreEvent: NONE=0, TRUE=1, PERF_CNT_0=5, BROADCAST_0=107
# CHECK: AIE2P MemEvent: NONE=0, WATCHPOINT_0=16
# CHECK: Edge case: Same event name, different values across archs
# CHECK: USER_EVENT_0: AIE1.Shim=124, AIE2.Shim=126
# CHECK: DMA_S2MM_0_FINISHED_BD: AIE1.Mem=25, AIE1.Shim=16
# CHECK: Edge case: Arch-specific events
# CHECK: AIE1.MemEvent has WATCHPOINT_0
# CHECK: AIE2.MemEvent has WATCHPOINT_0
# CHECK: AIE2 has MemTileEvent, AIE1 does not
# CHECK: AIE2P has MemTileEvent, AIE1 does not
# CHECK: Edge case: Events not present in arch raise AttributeError
# CHECK: AIE1.CoreEvent.EDGE_DETECTION_EVENT_0 does not exist
# CHECK: AIE2.CoreEvent.SRS_SATURATE does not exist
# CHECK: All tests passed!

import aie.utils.trace.events.aie as aie1
import aie.utils.trace.events.aie2 as aie2
import aie.utils.trace.events.aie2p as aie2p


def test_arch(name, core_cls, mem_cls, shim_cls, mem_tile_cls=None):
    """Test event enums for a given architecture"""
    print(f"Testing {name}")

    # Test CoreEvent
    assert core_cls.NONE.value == 0 and core_cls.TRUE.value == 1
    for m in core_cls:
        assert isinstance(m.value, int) and isinstance(m.name, str)

    # Test MemEvent
    assert mem_cls.NONE.value == 0
    for m in mem_cls:
        assert isinstance(m.value, int) and isinstance(m.name, str)

    # Test ShimEvent
    assert shim_cls.NONE.value == 0
    for m in shim_cls:
        assert isinstance(m.value, int) and isinstance(m.name, str)

    # Test MemTileEvent (AIE2/AIE2P only)
    if mem_tile_cls:
        assert mem_tile_cls.NONE.value == 0
        for m in mem_tile_cls:
            assert isinstance(m.value, int) and isinstance(m.name, str)

    # Print sample values for verification
    if name == "AIE1":
        print(
            f"AIE1 CoreEvent: NONE={core_cls.NONE.value}, TRUE={core_cls.TRUE.value}, "
            f"PERF_CNT_0={core_cls.PERF_CNT_0.value}, BROADCAST_15={core_cls.BROADCAST_15.value}"
        )
        print(
            f"AIE1 MemEvent: NONE={mem_cls.NONE.value}, "
            f"DMA_S2MM_0_FINISHED_BD={mem_cls.DMA_S2MM_0_FINISHED_BD.value}"
        )
        print(
            f"AIE1 ShimEvent: NONE={shim_cls.NONE.value}, "
            f"DMA_S2MM_0_FINISHED_BD={shim_cls.DMA_S2MM_0_FINISHED_BD.value}"
        )
    elif name == "AIE2":
        print(
            f"AIE2 CoreEvent: NONE={core_cls.NONE.value}, TRUE={core_cls.TRUE.value}, "
            f"PERF_CNT_0={core_cls.PERF_CNT_0.value}, BROADCAST_0={core_cls.BROADCAST_0.value}"
        )
        print(
            f"AIE2 MemEvent: NONE={mem_cls.NONE.value}, WATCHPOINT_0={mem_cls.WATCHPOINT_0.value}, "
            f"DMA_S2MM_0_START_TASK={mem_cls.DMA_S2MM_0_START_TASK.value}"
        )
        print(
            f"AIE2 ShimEvent: NONE={shim_cls.NONE.value}, "
            f"DMA_S2MM_0_START_TASK={shim_cls.DMA_S2MM_0_START_TASK.value}"
        )
        print(
            f"AIE2 MemTileEvent: NONE={mem_tile_cls.NONE.value}, "
            f"CONFLICT_DM_BANK_0={mem_tile_cls.CONFLICT_DM_BANK_0.value}"
        )
    elif name == "AIE2P":
        print(
            f"AIE2P CoreEvent: NONE={core_cls.NONE.value}, TRUE={core_cls.TRUE.value}, "
            f"PERF_CNT_0={core_cls.PERF_CNT_0.value}, BROADCAST_0={core_cls.BROADCAST_0.value}"
        )
        print(
            f"AIE2P MemEvent: NONE={mem_cls.NONE.value}, WATCHPOINT_0={mem_cls.WATCHPOINT_0.value}"
        )


def test_edge_cases():
    """Test edge cases: same event names with different values, arch-specific events"""
    print("Edge case: Same event name, different values across archs")

    # Same event name across architectures - USER_EVENT_0 in ShimTileEvent has different values
    aie1_user0 = aie1.ShimTileEvent.USER_EVENT_0.value
    aie2_user0 = aie2.ShimTileEvent.USER_EVENT_0.value
    print(f"USER_EVENT_0: AIE1.Shim={aie1_user0}, AIE2.Shim={aie2_user0}")
    assert (
        aie1_user0 != aie2_user0
    ), "USER_EVENT_0 should have different values in AIE1 vs AIE2"

    # Same event name, different modules in same arch
    aie1_dma_mem = aie1.MemEvent.DMA_S2MM_0_FINISHED_BD.value
    aie1_dma_shim = aie1.ShimTileEvent.DMA_S2MM_0_FINISHED_BD.value
    print(f"DMA_S2MM_0_FINISHED_BD: AIE1.Mem={aie1_dma_mem}, AIE1.Shim={aie1_dma_shim}")
    assert (
        aie1_dma_mem != aie1_dma_shim
    ), "Same event in different modules should have different values"

    print("Edge case: Arch-specific events")

    # WATCHPOINT exists in all archs for MemEvent
    assert hasattr(aie1.MemEvent, "WATCHPOINT_0"), "AIE1 should have WATCHPOINT_0"
    print("AIE1.MemEvent has WATCHPOINT_0")
    assert hasattr(aie2.MemEvent, "WATCHPOINT_0"), "AIE2 should have WATCHPOINT_0"
    print("AIE2.MemEvent has WATCHPOINT_0")

    # MemTileEvent only exists in AIE2/AIE2P
    assert hasattr(aie1, "MemTileEvent"), "AIE1 module has MemTileEvent class"
    # But AIE1's MemTileEvent has no events (just pass)
    aie1_memtile_events = [e for e in aie1.MemTileEvent]
    assert len(aie1_memtile_events) == 0, "AIE1 MemTileEvent should be empty"
    print("AIE2 has MemTileEvent, AIE1 does not")

    # AIE2/AIE2P have actual MemTileEvent events
    assert hasattr(aie2, "MemTileEvent"), "AIE2 should have MemTileEvent"
    aie2_memtile_events = [e for e in aie2.MemTileEvent]
    assert len(aie2_memtile_events) > 0, "AIE2 MemTileEvent should have events"

    assert hasattr(aie2p, "MemTileEvent"), "AIE2P should have MemTileEvent"
    aie2p_memtile_events = [e for e in aie2p.MemTileEvent]
    assert len(aie2p_memtile_events) > 0, "AIE2P MemTileEvent should have events"
    print("AIE2P has MemTileEvent, AIE1 does not")

    print("Edge case: Events not present in arch raise AttributeError")

    # EDGE_DETECTION_EVENT exists in AIE2 CoreEvent but not AIE1 CoreEvent
    assert hasattr(
        aie2.CoreEvent, "EDGE_DETECTION_EVENT_0"
    ), "AIE2 should have EDGE_DETECTION_EVENT_0"
    try:
        _ = aie1.CoreEvent.EDGE_DETECTION_EVENT_0
        assert False, "AIE1.CoreEvent.EDGE_DETECTION_EVENT_0 should not exist"
    except AttributeError:
        print("AIE1.CoreEvent.EDGE_DETECTION_EVENT_0 does not exist")

    # SRS_SATURATE exists in AIE1 but not AIE2
    assert hasattr(aie1.CoreEvent, "SRS_SATURATE"), "AIE1 should have SRS_SATURATE"
    try:
        _ = aie2.CoreEvent.SRS_SATURATE
        assert False, "AIE2.CoreEvent.SRS_SATURATE should not exist"
    except AttributeError:
        print("AIE2.CoreEvent.SRS_SATURATE does not exist")


if __name__ == "__main__":
    test_arch("AIE1", aie1.CoreEvent, aie1.MemEvent, aie1.ShimTileEvent)
    test_arch(
        "AIE2", aie2.CoreEvent, aie2.MemEvent, aie2.ShimTileEvent, aie2.MemTileEvent
    )
    test_arch(
        "AIE2P",
        aie2p.CoreEvent,
        aie2p.MemEvent,
        aie2p.ShimTileEvent,
        aie2p.MemTileEvent,
    )
    test_edge_cases()
    print("All tests passed!")
