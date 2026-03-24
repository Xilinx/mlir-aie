# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logger = logging.getLogger(__name__)

from aie.dialects.aie import (
    packetflow,
    WireBundle,
    trace,
    trace_mode,
    trace_event,
    trace_packet,
    trace_port,
    trace_start,
    trace_stop,
    trace_start_config,
    trace_host_config,
    TraceMode,
    TracePacketType,
    TraceShimRouting,
    DMAChannelDir,
    get_target_model,
)
from aie.dialects.aiex import (
    npu_write32,
    npu_writebd,
    npu_maskwrite32,
    npu_address_patch,
    npu_sync,
)
from .events import (
    BasePortEvent,
    GenericEvent,
    PortEvent,
    CoreEvent,
    MemEvent,
    ShimTileEvent,
    MemTileEvent,
    MemTilePortEvent,
    PacketType,
)

# Globally defined constants
direction_s2mm = 0
direction_mm2s = 1


# Push specified bd_id to task queue
def push_bd_to_task_queue(shim, bd_id, direction, channel, enable_token):
    task_queue_addr = 0x0
    if direction == direction_s2mm:
        if channel == 0:
            task_queue_addr = 0x1D204
        else:
            task_queue_addr = 0x1D20C
    else:  # direction_mm2s
        if channel == 0:
            task_queue_addr = 0x1D214
        else:
            task_queue_addr = 0x1D21C

    # Add bd to task queue to begin transaction
    npu_write32(
        column=int(shim.col),
        row=int(shim.row),
        address=task_queue_addr,
        value=((enable_token & 0x1) << 31) | bd_id,
    )


# Configure shim tile's DMA for tracing.
# This configures the shim tile / bd to process a specficic `packet id`
# and `packet type`. It also configures the address patch.
def configure_shimtile_dma_aie2(
    shim,
    channel=1,
    direction=direction_s2mm,
    bd_id=13,
    ddr_id=4,  # ddr_id (0,1,2,3,4) -> xrt bo/ grp id (3,4,5,6,7)
    size=8192,  # in words (32-bits)
    offset=0,  # in bytes
    enable_token=0,
    enable_packet=1,  # valid for mm2s xfer only
    packet_id=0,  # for mm2s xfer
    packet_type=PacketType.CORE,  # for mm2s xfer
    shim_burst_length=64,
):

    dev = shim.parent.attributes["device"]
    tm = get_target_model(dev)

    # Shim has to be a shim tile
    assert tm.is_shim_noc_tile(shim.col, shim.row)

    # configure_shimtile_bd_aie2(shim, channel, bd_id, ddr_id, size, offset, 1, 0, 0)
    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
    # out to host DDR memory
    npu_writebd(
        bd_id=bd_id,
        buffer_length=size,  # buffer length in words (32 bits)
        buffer_offset=offset,  # offset in bytes
        enable_packet=enable_packet,  # valid for mm2s xfer only, direction=1
        out_of_order_id=0,
        packet_id=packet_id,  # used for mm2s xfer, direction=1
        packet_type=packet_type,  # used for mm2s xfer, direction=1
        column=int(shim.col),
        d0_size=0,
        d0_stride=0,
        d0_zero_after=0,
        d0_zero_before=0,
        d1_size=0,
        d1_stride=0,
        d1_zero_after=0,
        d1_zero_before=0,
        d2_size=0,
        d2_stride=0,
        d2_zero_after=0,
        d2_zero_before=0,
        burst_length=shim_burst_length,
        iteration_current=0,
        iteration_size=0,
        iteration_stride=0,
        lock_acq_enable=0,
        lock_acq_id=0,
        lock_acq_val=0,
        lock_rel_id=0,
        lock_rel_val=0,
        next_bd=0,
        row=0,
        use_next_bd=0,
        valid_bd=1,
    )
    addr = (int(shim.col) << tm.get_column_shift()) | (0x1D004 + bd_id * 0x20)
    npu_address_patch(addr=addr, arg_idx=ddr_id, arg_plus=offset)

    ctrl_addr = 0x0
    if direction == direction_s2mm:
        if channel == 0:
            ctrl_addr = 0x1D200
        else:
            ctrl_addr = 0x1D208
    else:  # direction_mm2s
        if channel == 0:
            ctrl_addr = 0x1D210
        else:
            ctrl_addr = 0x1D218

    # Set controller id to match autorouted tilecontrol->south0 which is 15 (xF)
    # This is needed so tct tokens are properly routed for us to sync on.
    npu_maskwrite32(
        column=int(shim.col),
        row=int(shim.row),
        address=ctrl_addr,
        value=0xF00,  # pkt_id = 15
        mask=0x1F00,
    )

    push_bd_to_task_queue(
        shim=shim,
        bd_id=bd_id,
        direction=direction,
        channel=channel,
        enable_token=enable_token,
    )


# default packet_id = 30 for broadcast packet to all tiles to trace/ ctrl packet
# We do not choose id = 31 in case that's might be chosen for a special purpose
# return packet_id starts at 29 and goes down. We run into potential conflict when we get
# to id = 15 since that's used for routing of tct token in shim stream switch to
# axi interface (south 0). So in practice, we support 14 targets.
def configure_packet_ctrl_flow(
    tiles_to_trace,
    shim,
    tile2shim_pkt_id=28,  # WARNING: Careful not to change. Must match stream id for control packet
    # shim2tile_pkt_id = 16,
    shim2tile_pkt_id=29,  # WARNING: Careful not to change. Used by config_ctrl_pkts_aie
):

    exist_traces = []
    for i in range(len(tiles_to_trace)):

        if tiles_to_trace[i] not in exist_traces:

            exist_traces.append(tiles_to_trace[i])

            # id 14 - tile -> shim
            packetflow(
                tile2shim_pkt_id,
                tiles_to_trace[i],
                WireBundle.TileControl,
                0,
                dests={"dest": shim, "port": WireBundle.DMA, "channel": 0},
                keep_pkt_header=True,
            )

            # id 15 - shim -> tile
            packetflow(
                shim2tile_pkt_id,
                shim,
                WireBundle.DMA,
                0,
                dests={
                    "dest": tiles_to_trace[i],
                    "port": WireBundle.TileControl,
                    "channel": 0,
                },
            )


# Configure shim for control packets.
#
# We configure the input shim (mm2s) to read num_pkts control commands where we
# assume control command consists of 1x 32b word. We use ddr_id=6 as the input XRT buffer
# which is not used in simple 1 in/ 1 out and 2 in/ 1 out designs.
#
# On the output shim (s2mm), we write the control packet responses to the same
# XRT buffer as trace (ddr_id=4) and write the results after the end of the trace buffer.
# Here, responses are compsed of 2x 32-bit word per control command.
#
# NOTE:
# We cannot batch control commands right now. Each control command has to be a separate
# transfer. Hence, we need to sync between each config.
#
# Only supports one target so far
# * need packet router at shim to use mask to allow multiple packet_ids
def config_ctrl_pkts_aie(
    tiles_to_trace,
    shim,
    shim2tile_pkt_id=29,  # WARNING: Careful not to change. Used by configure_packet_ctrl_flow
    output_offset=0,  # needs offset from trace buffer
    num_pkts=1,
    channel=0,
    mm2s_bd_id=14,  # TODO: default bd ids (hopefully doesn't conflict with other shim cfg)
    s2mm_bd_id=15,  # TODO: default bd ids (hopefully doesn't conflict with other shim cfg)
):

    for i in range(num_pkts):
        # config mm2s shim
        configure_shimtile_dma_aie2(
            shim=shim,
            channel=channel,
            direction=direction_mm2s,
            bd_id=mm2s_bd_id,
            ddr_id=3,  # group_id(6)
            size=1,  # 1x 32b word
            offset=(i * 4),  # bytes
            packet_id=shim2tile_pkt_id,
            packet_type=0,  # no used in same way as trace to identiyf source
            enable_token=1,
        )

        npu_sync(
            column=int(shim.col),
            column_num=1,
            row=0,
            direction=direction_mm2s,
            channel=channel,
        )  # input

        # config s2mm shim
        configure_shimtile_dma_aie2(
            shim=shim,
            channel=channel,
            direction=direction_s2mm,
            bd_id=s2mm_bd_id,
            # ddr_id=3, # group_id(6)
            ddr_id=4,  # group_id(7)
            size=2,  # 2 32b words, pkt header + 1 word data
            offset=output_offset + (i * 8),  # bytes
            # packet_id=tile2shim_pkt_id, # NOT USED
            packet_type=0,  # not used in same way as trace to identify source
            enable_token=1,
        )

        npu_sync(
            column=int(shim.col),
            column_num=1,
            row=0,
            direction=direction_s2mm,
            channel=channel,
        )  # output


# ============================================================================
# Declarative Trace API
# ============================================================================

# Module-level storage for trace names configured by configure_trace()
_configured_trace_names = []


def _get_packet_type_for_tile(tile_op, is_mem_trace=False):
    """Determine the TracePacketType based on tile type."""
    if tile_op.is_core_tile():
        return TracePacketType.Mem if is_mem_trace else TracePacketType.Core
    elif tile_op.is_mem_tile():
        return TracePacketType.MemTile
    elif tile_op.is_shim_tile():
        return TracePacketType.ShimTile
    else:
        raise ValueError(f"Unknown tile type for {tile_op}")


def _get_default_events_for_tile(tile_op, is_mem_trace=False):
    """Get default trace events for a tile type."""
    if tile_op.is_core_tile():
        if is_mem_trace:
            # Core memory trace defaults
            return [
                MemEvent.DMA_S2MM_0_START_TASK,
                MemEvent.DMA_MM2S_0_START_TASK,
                MemEvent.CONFLICT_DM_BANK_0,
                MemEvent.CONFLICT_DM_BANK_1,
                MemEvent.CONFLICT_DM_BANK_2,
                MemEvent.CONFLICT_DM_BANK_3,
                MemEvent.EDGE_DETECTION_EVENT_0,
                MemEvent.EDGE_DETECTION_EVENT_1,
            ]
        else:
            # Core tile trace defaults
            return [
                CoreEvent.INSTR_EVENT_0,
                CoreEvent.INSTR_EVENT_1,
                CoreEvent.INSTR_VECTOR,
                CoreEvent.MEMORY_STALL,
                CoreEvent.STREAM_STALL,
                CoreEvent.LOCK_STALL,
                PortEvent(
                    CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True
                ),  # DMA ch0 in
                PortEvent(
                    CoreEvent.PORT_RUNNING_1, WireBundle.DMA, 0, False
                ),  # DMA ch0 out
            ]
    elif tile_op.is_mem_tile():
        return [
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True
            ),  # DMA ch0 in
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_1, WireBundle.DMA, 1, True
            ),  # DMA ch1 in
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_2, WireBundle.DMA, 2, True
            ),  # DMA ch2 in
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_3, WireBundle.DMA, 3, True
            ),  # DMA ch3 in
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_4, WireBundle.DMA, 0, False
            ),  # DMA ch0 out
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_5, WireBundle.DMA, 1, False
            ),  # DMA ch1 out
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_6, WireBundle.DMA, 2, False
            ),  # DMA ch2 out
            MemTilePortEvent(
                MemTileEvent.PORT_RUNNING_7, WireBundle.DMA, 3, False
            ),  # DMA ch3 out
        ]
    elif tile_op.is_shim_tile():
        return [
            ShimTileEvent.DMA_S2MM_0_START_TASK,
            ShimTileEvent.DMA_S2MM_1_START_TASK,
            ShimTileEvent.DMA_MM2S_0_START_TASK,
            ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
            ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
            ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
            ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
            ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
        ]
    else:
        return []


def configure_trace(
    tiles_to_trace,
    start_broadcast=15,
    stop_broadcast=14,
    coretile_events=None,
    coremem_events=None,
    memtile_events=None,
    shimtile_events=None,
):
    """Generate aie.trace ops for a list of tiles.

    This function emits declarative aie.trace operations that define what
    events to trace.

    Args:
        tiles_to_trace: List of tile operations to configure tracing for.
        start_broadcast: Broadcast channel for trace start event (default: 15).
        stop_broadcast: Broadcast channel for trace stop event (default: 14).
        coretile_events: List of events for core tile tracing (max 8).
        coremem_events: List of events for core memory tracing (max 8).
        memtile_events: List of events for mem tile tracing (max 8).
        shimtile_events: List of events for shim tile tracing (max 8).
    """
    _configured_trace_names.clear()

    if not tiles_to_trace:
        return

    packet_id = 1
    seen_core_tiles = set()

    for tile_op in tiles_to_trace:
        # Determine if this is a core tile memory trace (second occurrence)
        is_mem_trace = False
        if tile_op.is_core_tile():
            tile_key = id(tile_op)  # Use object identity for tracking
            if tile_key in seen_core_tiles:
                is_mem_trace = True
            else:
                seen_core_tiles.add(tile_key)

        # Generate unique trace name based on tile type
        if tile_op.is_core_tile():
            trace_type = "mem" if is_mem_trace else "core"
        elif tile_op.is_mem_tile():
            trace_type = "memtile"
        elif tile_op.is_shim_tile():
            trace_type = "shim"
        else:
            raise ValueError(f"Unknown tile type for tracing: {tile_op}")

        trace_name = f"trace_{trace_type}_{packet_id}"

        # Get events for this tile type
        if tile_op.is_core_tile():
            events = coremem_events if is_mem_trace else coretile_events
            if events is None:
                events = _get_default_events_for_tile(tile_op, is_mem_trace)
        elif tile_op.is_mem_tile():
            events = memtile_events
            if events is None:
                events = _get_default_events_for_tile(tile_op)
        elif tile_op.is_shim_tile():
            events = shimtile_events
            if events is None:
                events = _get_default_events_for_tile(tile_op)
        else:
            raise ValueError(f"Unknown tile type for tracing: {tile_op}")

        # Validate events - wrap in GenericEvent if not already
        events = [e if isinstance(e, GenericEvent) else GenericEvent(e) for e in events]

        # Get packet type
        packet_type = _get_packet_type_for_tile(tile_op, is_mem_trace)
        is_core_trace = tile_op.is_core_tile() and not is_mem_trace

        # Pad events to 8 with NONE events
        if len(events) > 8:
            raise RuntimeError(
                f"At most 8 events can be traced at once, have {len(events)}."
            )
        if tile_op.is_core_tile():
            none_event = MemEvent.NONE if is_mem_trace else CoreEvent.NONE
        elif tile_op.is_mem_tile():
            none_event = MemTileEvent.NONE
        elif tile_op.is_shim_tile():
            none_event = ShimTileEvent.NONE
        else:
            none_event = CoreEvent.NONE
        padded_events = (list(events) + [none_event] * 8)[:8]

        # Collect and validate port events - multiple events can share a slot
        # (e.g., PORT_RUNNING_0 and PORT_TLAST_0 both use slot 0) but must have
        # the same port configuration.
        port_configs = {}  # slot -> (port, channel, direction, event_name)
        for event in padded_events:
            if isinstance(event, BasePortEvent):
                slot = event.slot
                config = (event.port, event.channel, event.direction)
                if slot in port_configs:
                    prev_config, prev_name = port_configs[slot]
                    if prev_config != config:
                        raise ValueError(
                            f"Conflicting port configurations for slot {slot}: "
                            f"{prev_name} uses {prev_config}, but "
                            f"{event.code.name} uses {config}. "
                            f"Events sharing a slot must monitor the same port."
                        )
                else:
                    port_configs[slot] = (config, event.code.name)

        # Generate the aie.trace op
        @trace(tile_op, trace_name)
        def trace_body():
            if is_core_trace:
                trace_mode(TraceMode.EventTime)
            trace_packet(packet_id, packet_type)

            for event in padded_events:
                trace_event(event)

            # Emit one trace_port per unique slot
            for slot, (config, _) in port_configs.items():
                port, channel, direction = config
                trace_port(slot, port, channel, direction)

            # All tiles use broadcast start/stop. Trace lowering pass
            # handles special case where a traced shim tile is also
            # the trace destination and configures timer control with
            # USER_EVENT_1
            trace_start(broadcast=start_broadcast)
            trace_stop(broadcast=stop_broadcast)

        _configured_trace_names.append(trace_name)
        packet_id += 1


def configure_trace_output(
    trace_size=8192,
    ddr_id=4,
    routing="single",
    trace_after_last_tensor=False,
):
    """Configure trace output buffer and emit start_config ops.

    This function should be called inside a runtime_sequence context
    to configure how trace data is collected and routed to host memory.

    Args:
        trace_size: Trace buffer size in bytes. Default is 8192.
        ddr_id: DDR buffer index (0-4) mapping to XRT group_id (3-7).
                Default is 4 (group_id 7).
        routing: Shim routing strategy - "single" (all to column 0) or
                 "per_column" (each column to its own shim).
        trace_after_last_tensor: If True, append trace data after the last
                                 runtime_sequence tensor argument. Only valid
                                 with routing="single".
    """
    # Map string routing to enum
    if routing == "single":
        routing_attr = TraceShimRouting.Single
    elif routing == "per_column":
        routing_attr = TraceShimRouting.PerColumn
    else:
        raise ValueError(f"Unknown routing strategy: {routing}.")

    # Emit host_config op
    trace_host_config(
        buffer_size=trace_size,
        arg_idx=ddr_id,
        routing=routing_attr,
        trace_after_last_tensor=trace_after_last_tensor,
    )

    # Emit start_config for each configured trace
    for trace_name in _configured_trace_names:
        trace_start_config(trace_name)
