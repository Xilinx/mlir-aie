# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

"""
Test Tile validation - error checking for type mismatches and duplicate COMPUTE tiles.
"""

import pytest
import numpy as np
from aie.iron import Worker, Runtime, ObjectFifo, Program
from aie.iron.device import Tile, NPU2
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import device as aie_device, AIEDevice


def test_worker_rejects_wrong_tile_type():
    """Worker should reject non-COMPUTE tile types."""

    def kernel():
        pass

    # Should reject MEMORY type
    with pytest.raises(ValueError, match="Worker requires Tile.COMPUTE"):
        Worker(kernel, placement=Tile(tile_type=Tile.MEMORY))

    # Should reject SHIM type
    with pytest.raises(ValueError, match="Worker requires Tile.COMPUTE"):
        Worker(kernel, placement=Tile(tile_type=Tile.SHIM))

    # Should accept COMPUTE type
    worker = Worker(kernel, placement=Tile(tile_type=Tile.COMPUTE))
    assert worker.tile.tile_type == Tile.COMPUTE

    # Should accept None and set to COMPUTE
    worker = Worker(kernel, placement=Tile())
    assert worker.tile.tile_type == Tile.COMPUTE


def test_tile_type_coordinate_mismatch():
    """Tile type should match device coordinates."""
    from aie.iron.device import NPU1Col1

    def kernel():
        pass

    dev = NPU1Col1()

    with mlir_mod_ctx() as ctx:

        @aie_device(AIEDevice.npu1_1col)
        def device_body():
            # Row 0 is SHIM, but user claims COMPUTE - should error
            worker = Worker(kernel, placement=Tile(0, 0, tile_type=Tile.COMPUTE))

            with pytest.raises(ValueError, match="Tile type mismatch"):
                dev.resolve_tile(worker.tile)


def test_duplicate_compute_tile_error():
    """Same COMPUTE tile cannot be allocated twice."""
    from aie.iron.device import NPU1Col1

    def kernel1():
        pass

    def kernel2():
        pass

    dev = NPU1Col1()

    with mlir_mod_ctx() as ctx:

        @aie_device(AIEDevice.npu1_1col)
        def device_body():
            # Two workers with same coordinates
            worker1 = Worker(kernel1, placement=Tile(0, 2))
            worker2 = Worker(kernel2, placement=Tile(0, 2))

            # First worker succeeds
            dev.resolve_tile(worker1.tile)

            # Second worker should fail - duplicate COMPUTE tile
            with pytest.raises(ValueError, match="already allocated"):
                dev.resolve_tile(worker2.tile)


def test_memory_shim_tiles_can_merge():
    """MEMORY and SHIM tiles CAN be reused (unlike COMPUTE)."""
    from aie.iron.device import NPU1Col1

    dev = NPU1Col1()

    with mlir_mod_ctx() as ctx:

        @aie_device(AIEDevice.npu1_1col)
        def device_body():
            # Multiple MEMORY tiles with same coordinates - OK
            mem1 = Tile(0, 1, tile_type=Tile.MEMORY)
            mem2 = Tile(0, 1, tile_type=Tile.MEMORY)
            dev.resolve_tile(mem1)
            dev.resolve_tile(mem2)  # Should not raise

            # Multiple SHIM tiles with same coordinates - OK
            shim1 = Tile(0, 0, tile_type=Tile.SHIM)
            shim2 = Tile(0, 0, tile_type=Tile.SHIM)
            dev.resolve_tile(shim1)
            dev.resolve_tile(shim2)  # Should not raise


def test_invalid_tile_type_string():
    """Invalid tile_type string should be rejected."""

    with pytest.raises(ValueError, match="Invalid tile_type"):
        Tile(tile_type="invalid")

    with pytest.raises(ValueError, match="Invalid tile_type"):
        Tile(tile_type="core")  # Should be "compute" not "core"


def test_objectfifo_link_rejects_shim():
    """ObjectFifoLink should reject SHIM tile type (only MEMORY or COMPUTE allowed)."""

    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of_in = ObjectFifo(n_ty, name="in")

    # Should reject SHIM type for forward()
    with pytest.raises(
        ValueError, match="ObjectFifoLink requires Tile.MEMORY or Tile.COMPUTE"
    ):
        of_out = of_in.cons().forward(placement=Tile(tile_type=Tile.SHIM))

    # Should accept MEMORY (default)
    of_out = of_in.cons().forward()  # OK - defaults to MEMORY

    # Should accept COMPUTE (special case)
    of_out2 = of_in.cons().forward(placement=Tile(0, 2))  # OK if (0,2) is COMPUTE


if __name__ == "__main__":
    test_worker_rejects_wrong_tile_type()
    test_tile_type_coordinate_mismatch()
    test_duplicate_compute_tile_error()
    test_memory_shim_tiles_can_merge()
    test_invalid_tile_type_string()
    test_objectfifo_link_rejects_shim()
