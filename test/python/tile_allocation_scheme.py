# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.iron import Program, Worker

from aie.iron.device import NPU1, Tile

# =============================================================================
# CHECK: {allocation_scheme = "bank-aware"}

my_worker = Worker(None, tile=Tile(1, 2, allocation_scheme="bank-aware"))


def runtime_sequence():
    pass


my_program = Program(NPU1(), runtime_sequence, arg_types=[], workers=[my_worker])

module = my_program.resolve_program()

print(module)

# =============================================================================
# CHECK-NOT: allocation_scheme =

my_worker = Worker(None)


def runtime_sequence():
    pass


my_program = Program(NPU1(), runtime_sequence, arg_types=[], workers=[my_worker])

module = my_program.resolve_program()

print(module)

# =============================================================================
# CHECK: {allocation_scheme = "basic-sequential"}

my_worker = Worker(
    None,
    tile=Tile(1, 2, allocation_scheme="bank-aware"),
    allocation_scheme="basic-sequential",
)


def runtime_sequence():
    pass


my_program = Program(NPU1(), runtime_sequence, arg_types=[], workers=[my_worker])

module = my_program.resolve_program()

print(module)

# =============================================================================
# CHECK: {allocation_scheme = "bank-aware"}

my_worker = Worker(None, tile=Tile(1, 2), allocation_scheme="bank-aware")


def runtime_sequence():
    pass


my_program = Program(NPU1(), runtime_sequence, arg_types=[], workers=[my_worker])

module = my_program.resolve_program()

print(module)

# =============================================================================
# CHECK: {allocation_scheme = "basic-sequential"}

my_worker = Worker(None, allocation_scheme="basic-sequential")


def runtime_sequence():
    pass


my_program = Program(NPU1(), runtime_sequence, arg_types=[], workers=[my_worker])

module = my_program.resolve_program()

print(module)
