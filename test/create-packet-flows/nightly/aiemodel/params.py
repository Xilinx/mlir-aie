#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Model constants and test knobs, shared by every module."""

PARAMS = dict(
    # numArbiters, numMselsPerArbiter: AIECreatePathFindFlows.cpp
    arbiters=6,
    msels=4,
    # AIETargetModel::getNumSlaveSlots, getMaxPacketId
    rule_slots=4,
    max_id=31,
    # pktHeaderBytes, maxBDSteps:
    # AIEStreamDependencyAnalysis.cpp
    header_bytes=4,
    max_bd_steps=1024,
    # stepBudget in planArbiters and unroutableArbiters, budget in
    # runOnPacketFlow's hold-cycle search: AIECreatePathFindFlows.cpp;
    # maxWalkSearches in StreamConflicts::holdCycle
    plan_budget=100000,
    clique_budget=100000,
    hold_budget=256,
    walk_searches=1024,
    # Designs per tier and per aie-opt run.
    routable=170,
    unroutable=40,
    unknown=50,
    batch=40,
    # Ratchets on the router's output.
    min_routed_rate=1.0,
    max_hop_ratio=1.10,
    max_amsels=dict(npu1=19.5, npu2=19.5, xcvc1902=21.0),
    max_low_priority=0.05,
    max_ms_per_design=dict(npu1=250, npu2=250, xcvc1902=600),
    # Generator knobs: share of routable designs given pre-placed switchbox
    # configuration, and parallel aie-opt runs.
    fixed_rate=0.25,
    jobs=4,
)
# Ratchet on the route space the test reaches: values hit per dimension
# (route_space, space_domain). Raise as the generator reaches more.
SPACE_FLOORS = {}

FAMILIES = {
    "npu1": ("npu1_1col", "npu1_2col", "npu1_3col", "npu1"),
    "npu2": ("npu2_1col", "npu2_3col", "npu2_4col", "npu2"),
}
DEVICE_IDS = {
    1: "xcvc1902",
    4: "npu1",
    5: "npu1_1col",
    6: "npu1_2col",
    7: "npu1_3col",
    8: "npu2",
    **{9 + k: f"npu2_{k + 1}col" for k in range(7)},
}
