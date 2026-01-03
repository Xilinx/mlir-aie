# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from aie.utils.parse_trace import parse_trace
import json


class TraceConfig:
    DEFAULT_TRACE_BUFFER_INDEX = 4

    def __init__(
        self,
        trace_size: int,
        trace_file: str = "trace.txt",
        trace_after_last_tensor: bool = False,
        enable_ctrl_pkts: bool = False,
        last_tensor_shape=None,
        last_tensor_dtype=None,
    ):
        if trace_size <= 0:
            raise ValueError(f"Invalid trace size: {trace_size}")
        self.trace_size = trace_size
        self.trace_file = trace_file
        self.trace_after_last_tensor = trace_after_last_tensor
        self.enable_ctrl_pkts = enable_ctrl_pkts
        self.last_tensor_shape = last_tensor_shape
        self.last_tensor_dtype = last_tensor_dtype

    # checks # of bits. Odd number returns a 1. Even returns 0.
    @staticmethod
    def parity(x):
        return x.bit_count() & 1

    # create control packet
    @staticmethod
    def create_ctrl_pkt(
        operation,
        beats,
        addr,
        ctrl_pkt_read_id=28,  # global id used for all ctrl packet reads
        # WARNING: this needs to match the packet id used in packetflow/.py
    ):
        header = (ctrl_pkt_read_id << 24) | (operation << 22) | (beats << 20) | addr
        header |= (0x1 ^ TraceConfig.parity(header)) << 31
        return header

    @staticmethod
    def extract_tile(data):
        col = (data >> 21) & 0x7F
        row = (data >> 16) & 0x1F
        pkt_type = (data >> 12) & 0x3
        pkt_id = data & 0x1F
        return (col, row, pkt_type, pkt_id)

    def write_trace(self, trace):
        out_str = "\n".join(f"{i:0{8}x}" for i in trace if i != 0)
        with open(self.trace_file, "w") as f:
            f.write(out_str)

    def read_trace(self):
        with open(self.trace_file, "r") as f:
            trace_data = [int(line.strip(), 16) for line in f if line.strip()]
        return np.array(trace_data, dtype=np.uint32)

    def trace_to_json(self, mlir_file: str, output_name: str = "trace.json"):
        """Wrapper over parse_trace.py utility."""
        trace_buffer = self.read_trace()

        with open(mlir_file, "r") as f:
            mlir_module_str = f.read()

        trace_events = parse_trace(trace_buffer, mlir_module_str)

        with open(output_name, "w") as f:
            json.dump(trace_events, f, indent=2)
