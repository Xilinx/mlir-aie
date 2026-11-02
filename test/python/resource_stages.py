# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from pathlib import Path

import aie.dialects.aie  # noqa: F401  registers the aie dialect
import aie.dialects.aiex  # noqa: F401  registers the aiex dialect
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager

DESIGN = Path(__file__).resolve().parents[1] / "resource-stages/resource_stages.mlir"

with Context() as ctx, Location.unknown():
    module = Module.parse(DESIGN.read_text())
    PassManager.parse("builtin.module(aie-resource-stages)").run(module.operation)
    text = str(module)
    # CHECK: placed True
    print("placed", "logical_tile" not in text)
    # CHECK: addressed True
    print("addressed", "address =" in text)
    # CHECK: routed True
    print("routed", "aie.switchbox" in text)
