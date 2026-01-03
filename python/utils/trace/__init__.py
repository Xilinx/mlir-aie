# SPDX-FileCopyrightText: Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .config import TraceConfig
from .event_enums import (
    CoreEvent,
    MemEvent,
    ShimTileEvent,
    MemTileEvent,
)
from .events import (
    GenericEvent,
    PortEvent,
    MemTilePortEvent,
    ShimTilePortEvent,
)
from .parse import parse_trace
from .port_events import (
    PacketType,
    PortEventCodes,
    MemTilePortEventCodes,
    ShimTilePortEventCodes,
    NUM_TRACE_TYPES,
)
from .setup import (
    configure_coremem_tracing_aie2,
    configure_coretile_tracing_aie2,
    configure_memtile_tracing_aie2,
    configure_shimtile_tracing_aie2,
    configure_packet_tracing_flow,
    configure_shim_trace_start_aie2,
    gen_trace_done_aie2,
    configure_packet_tracing_aie2,
    configure_simple_tracing_aie2,
    configure_packet_ctrl_flow,
    config_ctrl_pkts_aie,
)
from .utils import (
    parity,
    extract_tile,
    pack4bytes,
    create_ctrl_pkt,
    get_kernel_code,
    extract_buffers,
    get_cycles,
    get_cycles_summary,
    get_vector_time,
)
