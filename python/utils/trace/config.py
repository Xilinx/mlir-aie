# SPDX-FileCopyrightText: Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import json
from .parse import parse_trace
from .utils import parity, extract_tile


class TraceConfig:
    DEFAULT_TRACE_BUFFER_INDEX = 4
    DEFAULT_TRACE_FILE = "trace.txt"

    def __init__(
        self,
        trace_size: int,
        trace_file: str = DEFAULT_TRACE_FILE,
        ddr_id: int = DEFAULT_TRACE_BUFFER_INDEX,
        enable_ctrl_pkts: bool = False,
        last_tensor_shape=None,
        last_tensor_dtype=None,
    ):
        if trace_size <= 0:
            raise ValueError(f"Invalid trace size: {trace_size}")
        self.trace_size = trace_size
        self.trace_file = trace_file
        self.ddr_id = ddr_id
        self.enable_ctrl_pkts = enable_ctrl_pkts
        self.last_tensor_shape = last_tensor_shape
        self.last_tensor_dtype = last_tensor_dtype
        # Path to physical MLIR with lowered trace ops (set by NPUKernel)
        self.physical_mlir_path = None

    def __repr__(self) -> str:
        # Eval-faithful: only constructor kwargs.  ``eval(repr(cfg))``
        # round-trips to an equivalent fresh TraceConfig.  Post-run mutable
        # state (physical_mlir_path, last_tensor_*) lives in __str__.
        # Defaults are skipped to keep the rendering tight.
        bits = [f"trace_size={self.trace_size}"]
        if self.trace_file != self.DEFAULT_TRACE_FILE:
            bits.append(f"trace_file={self.trace_file!r}")
        if self.ddr_id != self.DEFAULT_TRACE_BUFFER_INDEX:
            bits.append(f"ddr_id={self.ddr_id}")
        if self.enable_ctrl_pkts:
            bits.append("enable_ctrl_pkts=True")
        return f"TraceConfig({', '.join(bits)})"

    def __str__(self) -> str:
        # Human-readable: starts with the eval-faithful repr, then appends
        # any post-run state someone debugging a trace would actually want
        # to see (where the lowered MLIR landed, what tensor shape was last
        # routed through this config).
        parts = [repr(self)]
        if self.physical_mlir_path is not None:
            parts.append(f"physical_mlir_path={self.physical_mlir_path}")
        if self.last_tensor_shape is not None:
            parts.append(f"last_tensor_shape={self.last_tensor_shape}")
        if self.last_tensor_dtype is not None:
            parts.append(f"last_tensor_dtype={self.last_tensor_dtype}")
        return parts[0] if len(parts) == 1 else "\n  ".join(parts)

    def write_trace(self, trace):
        # Strip only trailing zeros (unused buffer space). Internal zeros are
        # preserved -- they encode the gap between split DMA channel regions
        # when distribute-channels is active.
        end = len(trace)
        while end > 0 and trace[end - 1] == 0:
            end -= 1
        out_str = "\n".join(f"{i:0{8}x}" for i in trace[:end])
        with open(self.trace_file, "w") as f:
            f.write(out_str)

    def read_trace(self):
        with open(self.trace_file, "r") as f:
            trace_data = [int(line.strip(), 16) for line in f if line.strip()]
        buf = np.array(trace_data, dtype=np.uint32)
        # Pad back to the full trace buffer size. write_trace() strips
        # trailing zeros for compactness, but callers (especially with
        # distribute-channels) need the full buffer to index by channel
        # offset correctly.
        expected_words = self.trace_size // 4
        if len(buf) < expected_words:
            buf = np.pad(buf, (0, expected_words - len(buf)))
        return buf

    def trace_to_json(self, mlir_file: str, output_name: str = "trace.json"):
        """Wrapper over parse_trace.py utility."""
        trace_buffer = self.read_trace()

        with open(mlir_file, "r") as f:
            mlir_module_str = f.read()

        trace_events = parse_trace(trace_buffer, mlir_module_str)

        with open(output_name, "w") as f:
            json.dump(trace_events, f, indent=2)
