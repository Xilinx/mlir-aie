# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import numpy as np
from aie.utils.trace import parse_trace

trace = np.array([int(line, 16) for line in open(sys.argv[1]) if line.strip()], dtype=np.uint32)
events = parse_trace(trace, open(sys.argv[2]).read(), device_name="trace_dev")
counts = {name: sum(event.get("name") == name and event.get("ph") == "B" for event in events) for name in ("INSTR_EVENT_0", "INSTR_EVENT_1")}
print(counts)
assert counts == {"INSTR_EVENT_0": 1, "INSTR_EVENT_1": 1}
print("PASS!")
