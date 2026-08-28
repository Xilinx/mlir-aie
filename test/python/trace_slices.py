#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

# Test that a trace buffer shared by several aiex.configure'd designs is split
# by get_trace_slices(), and that each slice's events are read from the device
# that wrote it rather than from the union of all of them.

import subprocess
import tempfile
from pathlib import Path

from aie.utils.trace import get_trace_slices
from aie.utils.trace.parse import parse_mlir_trace_events

# Two designs, each traced, reached through aiex.configure. They trace
# different tiles and different events so a mixed-up attribution is visible.
SOURCE = """
module {
  aie.device(npu1_1col) @main {
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aiex.configure @dev_a {
        aiex.run @seq_a (%arg0) : (memref<64xi32>)
      }
      aiex.configure @dev_b {
        aiex.run @seq_b (%arg0) : (memref<64xi32>)
      }
    }
  }
  aie.device(npu1_1col) @dev_a {
    %shim_a = aie.tile(0, 0)
    %core_a = aie.tile(0, 2)
    aie.trace @a_trace(%core_a) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence @seq_a(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @a_trace
    }
  }
  aie.device(npu1_1col) @dev_b {
    %shim_b = aie.tile(0, 0)
    %core_b = aie.tile(0, 4)
    aie.trace @b_trace(%core_b) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence @seq_b(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 4096 : i32}
      aie.trace.start_config @b_trace
    }
  }
}
"""

PIPELINE = [
    "-aie-insert-trace-flows",
    "-aie-trace-to-config",
    "-aie-trace-pack-reg-writes",
    "-aie-inline-trace-config",
    "-aie-fuse-trace-buffers",
    "--aie-materialize-runtime-sequences",
    "-aie-resolve-address-patch-buffers",
]

with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "design.mlir"
    src.write_text(SOURCE)
    lowered = subprocess.run(
        ["aie-opt", str(src), *PIPELINE],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

slices = get_trace_slices(lowered)
for entry in slices:
    print(
        f"slice {entry['device']}/{entry['sequence']} "
        f"offset={entry['offset']} size={entry['size']}"
    )
# CHECK: slice dev_a/seq_a offset=0 size=8192
# CHECK: slice dev_b/seq_b offset=8192 size=4096

# Each slice decodes against its own device: dev_a traces row 2, dev_b row 4.
for entry in slices:
    pid_events, _, _ = parse_mlir_trace_events(lowered, None, entry["device"])
    tiles = sorted(pid_events[0].keys())
    events = [pid_events[0][t][0] for t in tiles]
    print(f"{entry['device']} tiles={tiles} first_events={events}")
# CHECK: dev_a tiles=['2,0'] first_events=[33]
# CHECK: dev_b tiles=['4,0'] first_events=[37]

# Parsing without naming a device merges both designs, which is what makes the
# device argument necessary once two of them share a buffer.
pid_events, _, _ = parse_mlir_trace_events(lowered, None, None)
print("merged tiles=", sorted(pid_events[0].keys()))
# CHECK: merged tiles= ['2,0', '4,0']
