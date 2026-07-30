#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

# Regression test: configure_trace() must emit a core memory trace for a
# tile that appears only ONCE in tiles_to_trace, as long as coremem_events
# is set. Before the fix, is_mem_trace only became True on a tile's SECOND
# occurrence in tiles_to_trace, so a single occurrence plus coremem_events
# silently produced no memory trace at all, matching how every real caller
# (including AMD's own programming_examples/basic/event_trace/aie_trace.py)
# lists a tile once.

from aie.dialects.aie import AIEDevice, device, tile
from aie.utils.trace.events import CoreEvent, MemEvent
import aie.utils.trace as trace_utils
from util import construct_and_print_module


# CHECK-LABEL: corememSingleOccurrence
# CHECK: aie.trace @trace_core_1
# CHECK:   aie.trace.packet type = core
# CHECK:   aie.trace.event <"INSTR_EVENT_0">
# CHECK:   aie.trace.start broadcast = 15
# CHECK:   aie.trace.stop broadcast = 14
# CHECK: aie.trace @trace_mem_2
# CHECK:   aie.trace.packet type = mem
# CHECK:   aie.trace.event <"DMA_S2MM_0_START_TASK">
# CHECK:   aie.trace.start broadcast = 15
# CHECK:   aie.trace.stop broadcast = 14
@construct_and_print_module
def corememSingleOccurrence():
    @device(AIEDevice.npu1_1col)
    def device_body():
        t = tile(0, 2)

        # `t` is listed exactly ONCE. The legacy trigger for a memory trace
        # was a tile's second occurrence in tiles_to_trace; passing
        # coremem_events must be sufficient on its own.
        trace_utils.configure_trace(
            [t],
            coretile_events=[CoreEvent.INSTR_EVENT_0],
            coremem_events=[MemEvent.DMA_S2MM_0_START_TASK],
        )
