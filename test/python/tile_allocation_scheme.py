# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.iron import Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4, Tile

# =============================================================================
# CHECK: {allocation_scheme = "bank-aware"}

my_worker = Worker(None, placement=Tile(1, 2, allocation_scheme="bank-aware"))

rt = Runtime()
with rt.sequence():
    rt.start(my_worker)

my_program = Program(NPU1Col4(), rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)

# =============================================================================
# CHECK-NOT: allocation_scheme =

my_worker = Worker(None)

rt = Runtime()
with rt.sequence():
    rt.start(my_worker)

my_program = Program(NPU1Col4(), rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)

# =============================================================================
# CHECK: {allocation_scheme = "basic-sequential"}

my_worker = Worker(
    None,
    placement=Tile(1, 2, allocation_scheme="bank-aware"),
    allocation_scheme="basic-sequential",
)

rt = Runtime()
with rt.sequence():
    rt.start(my_worker)

my_program = Program(NPU1Col4(), rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)

# =============================================================================
# CHECK: {allocation_scheme = "bank-aware"}

my_worker = Worker(None, placement=Tile(1, 2), allocation_scheme="bank-aware")

rt = Runtime()
with rt.sequence():
    rt.start(my_worker)

my_program = Program(NPU1Col4(), rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)

# =============================================================================
# CHECK: {allocation_scheme = "basic-sequential"}

my_worker = Worker(None, allocation_scheme="basic-sequential")

rt = Runtime()
with rt.sequence():
    rt.start(my_worker)

my_program = Program(NPU1Col4(), rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)
