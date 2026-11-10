# test_trace_ctrl_pkts.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""Reading the Trace_Status registers of the traced units with control packets.

The host side decodes the responses of the three reads in CTRL_PKT_READS, and
the design side routes and configures them. Overruns on hardware are covered by
npu-xrt/trace_overrun.
"""

import re

import numpy as np
import pytest
from aie.dialects.aie import AIEDevice, device, tile
from aie.dialects.aiex import runtime_sequence
from aie.extras.context import mlir_mod_ctx
from aie.iron import Program, Runtime, Worker
from aie.iron.device import NPU2, NPU2Col1
from aie.iron.configuration import DeviceConfiguration
from aie.utils.hostruntime.hostruntime import HostRuntime, HostRuntimeError
from aie.utils.trace import (
    TraceConfig,
    config_ctrl_pkts_aie,
    configure_packet_ctrl_flow,
)
from aie.utils.trace.utils import CTRL_PKT_CORE_ID, CTRL_PKT_MEM_ID

IDLE = 0x000
RUNNING = 0x100
OVERRUN = 0x300


def response(col, row, pkt_id, status):
    return [(col << 21) | (row << 16) | pkt_id, status]


def process(tmp_path, *responses):
    ctrl_buffer = np.zeros(4096, dtype=np.uint32)
    words = [word for r in responses for word in r]
    ctrl_buffer[: len(words)] = words
    config = TraceConfig(
        trace_size=8192, trace_file=str(tmp_path / "t.txt"), enable_ctrl_pkts=True
    )
    HostRuntime.process_trace(
        np.zeros(2048, dtype=np.uint32), ctrl_buffer.view(np.uint8), config
    )


def test_running_and_idle_units_pass(tmp_path):
    process(
        tmp_path,
        response(0, 2, CTRL_PKT_CORE_ID, RUNNING),
        response(0, 2, CTRL_PKT_MEM_ID, IDLE),
        response(0, 1, CTRL_PKT_CORE_ID, RUNNING),
        response(0, 0, CTRL_PKT_CORE_ID, RUNNING),
    )


def test_no_responses_pass(tmp_path):
    process(tmp_path)


@pytest.mark.parametrize(
    "col, row, pkt_id, unit",
    [
        (0, 2, CTRL_PKT_CORE_ID, "tile(0,2) core module"),
        (3, 5, CTRL_PKT_MEM_ID, "tile(3,5) memory module"),
        (1, 1, CTRL_PKT_CORE_ID, "tile(1,1) mem tile"),
        (2, 0, CTRL_PKT_CORE_ID, "tile(2,0) shim"),
    ],
)
def test_overrun_names_the_unit(tmp_path, col, row, pkt_id, unit):
    with pytest.raises(
        HostRuntimeError,
        match=r"overran in " + unit.replace("(", r"\(").replace(")", r"\)"),
    ):
        process(tmp_path, response(col, row, pkt_id, OVERRUN))


def test_overrun_lists_every_unit_that_overran(tmp_path):
    with pytest.raises(HostRuntimeError) as error:
        process(
            tmp_path,
            response(0, 2, CTRL_PKT_CORE_ID, OVERRUN),
            response(0, 3, CTRL_PKT_CORE_ID, RUNNING),
            response(0, 4, CTRL_PKT_CORE_ID, OVERRUN),
        )
    message = str(error.value)
    assert "tile(0,2) core module" in message
    assert "tile(0,4) core module" in message
    assert "tile(0,3)" not in message


def test_status_bits_other_than_state_are_ignored(tmp_path):
    # Mode is bits 2:0, only State, bits 9:8, tells an overrun.
    process(tmp_path, response(0, 2, CTRL_PKT_CORE_ID, 0x201))


def lower(tiles):
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            shim = tile(1, 0)
            traced = tiles()
            configure_packet_ctrl_flow(traced, shim)

            @runtime_sequence(
                np.ndarray[(8,), np.dtype[np.uint32]],
                np.ndarray[(2048,), np.dtype[np.uint8]],
            )
            def sequence(ctrl, trace):
                config_ctrl_pkts_aie(
                    traced, shim, ctrl_ddr_id=0, trace_ddr_id=1, output_offset=8192
                )

        assert ctx.module.operation.verify()
        return str(ctx.module)


def test_core_tiles_share_one_request_flow():
    mlir = lower(lambda: [tile(0, 2), tile(0, 3), tile(0, 2)])
    # One request id for all core tiles, a response id per module
    assert mlir.count("aie.packet_flow(29)") == 2
    assert mlir.count(f"aie.packet_flow({CTRL_PKT_CORE_ID})") == 2
    assert mlir.count(f"aie.packet_flow({CTRL_PKT_MEM_ID})") == 2
    # A core and memory module read, each answered by both tiles
    assert mlir.count("buffer_length = 4") == 2
    assert "buffer_offset = 8192" in mlir
    assert "buffer_offset = 8208" in mlir


def test_each_tile_type_gets_its_own_request_flow():
    mlir = lower(lambda: [tile(0, 2), tile(0, 1), tile(0, 0)])
    assert mlir.count("aie.packet_flow(29)") == 1
    assert mlir.count("aie.packet_flow(26)") == 1
    assert mlir.count("aie.packet_flow(25)") == 1


def test_responses_must_fit_the_host_buffer():
    with mlir_mod_ctx():

        @device(AIEDevice.npu2)
        def device_body():
            shim = tile(1, 0)
            tiles = [tile(0, row) for row in range(2, 6)]
            configure_packet_ctrl_flow(tiles, shim)

            @runtime_sequence(
                np.ndarray[(8,), np.dtype[np.uint32]],
                np.ndarray[(64,), np.dtype[np.uint8]],
            )
            def sequence(ctrl, trace):
                # 4 core tiles, 2 reads each, 8 bytes per tile: 64 bytes of
                # responses after the first 16 bytes of a 64 byte buffer.
                with pytest.raises(ValueError, match="need 64 bytes"):
                    config_ctrl_pkts_aie(
                        tiles,
                        shim,
                        ctrl_ddr_id=0,
                        trace_ddr_id=1,
                        output_offset=16,
                        host_buffer_bytes=64,
                    )


def emit_with_ctrl_pkts(runtime_types, sequence):
    worker = Worker(lambda: None, [], while_true=False, trace=1)
    runtime = Runtime(sequence, runtime_types)
    program = Program(NPU2(), runtime, workers=[worker])
    program.enable_trace(8192, workers=[worker], enable_ctrl_pkts=True)
    return str(program.resolve_program())


def patched_arg_indices(mlir):
    return sorted(set(int(i) for i in re.findall(r"arg_idx = (\d+)", mlir)))


def test_runtime_scalars_take_no_host_buffer_index():
    host_ty = np.ndarray[(16,), np.dtype[np.int32]]
    plain = emit_with_ctrl_pkts([host_ty], lambda host: None)
    scalar = emit_with_ctrl_pkts([host_ty, np.int32], lambda host, n: None)
    # The control buffer follows the one tensor, then the trace buffer
    assert patched_arg_indices(plain) == [1, 2]
    assert patched_arg_indices(scalar) == [1, 2]


def test_control_packets_reject_runtime_call():
    host_ty = np.ndarray[(16,), np.dtype[np.int32]]
    child = Runtime(lambda host: None, [host_ty], name="child")
    child_config = DeviceConfiguration("child_dev", NPU2Col1(), runtimes=[child])

    def entry_sequence(host):
        with child_config.configure():
            child.call(host)

    entry = Runtime(entry_sequence, [host_ty])
    main = DeviceConfiguration("main", NPU2Col1(), runtimes=[entry])
    child_config._enable_ctrl_pkts = True
    child_config._trace_size = 8192
    with pytest.raises(ValueError, match="Runtime.call"):
        Program.compose([main, child_config], entry=entry).resolve_program()


def test_responses_follow_both_trace_channels():
    host_ty = np.ndarray[(16,), np.dtype[np.int32]]
    mlir = emit_with_ctrl_pkts([host_ty], lambda host: None)
    # The first two trace_size bytes hold the trace, the responses come after
    assert "buffer_offset = 16384" in mlir


def test_host_splits_the_buffer_after_the_trace_region():
    trace_size = 64
    buffer = np.zeros(trace_size * 4, dtype=np.uint8)
    buffer[:trace_size] = 1  # first channel
    buffer[trace_size : trace_size * 2] = 2  # second channel
    buffer[trace_size * 2 : trace_size * 2 + 4] = 3  # first response word

    class Arg:
        def numpy(self):
            return buffer

    config = TraceConfig(trace_size=trace_size, enable_ctrl_pkts=True)
    trace, ctrl = HostRuntime.extract_trace_from_args([Arg()], config)
    assert trace.view(np.uint8).tolist() == [1] * trace_size
    assert ctrl.size == trace_size * 2
    assert ctrl[:4].tolist() == [3] * 4
