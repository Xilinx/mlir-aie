# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

"""
Test Tile validation: error checking for type mismatches and duplicate compute tiles.
"""

import pytest
import numpy as np
from aie.iron import Worker, ObjectFifo
from aie.iron.device import Tile, NPU2
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import device as aie_device, AIEDevice


def test_worker_rejects_wrong_tile_type():
    """Worker should reject non-compute tile types."""

    def kernel():
        pass

    # Should reject mem tile
    with pytest.raises(ValueError, match="Worker requires Tile.COMPUTE"):
        Worker(kernel, placement=Tile(tile_type=Tile.MEMORY))

    # Should reject shim tile
    with pytest.raises(ValueError, match="Worker requires Tile.COMPUTE"):
        Worker(kernel, placement=Tile(tile_type=Tile.SHIM))

    # Should accept compute tile
    worker = Worker(kernel, placement=Tile(tile_type=Tile.COMPUTE))
    assert worker.tile.tile_type == Tile.COMPUTE

    # Should accept None and set to compute
    worker = Worker(kernel, placement=Tile())
    assert worker.tile.tile_type == Tile.COMPUTE


def test_tile_type_coordinate_mismatch():
    """Tile type should match device coordinates."""

    def kernel():
        pass

    dev = NPU2()

    with mlir_mod_ctx() as ctx:

        @aie_device(AIEDevice.npu2)
        def device_body():
            # User tries to set worker on shim tile, should error
            worker = Worker(kernel, placement=Tile(0, 0, tile_type=Tile.COMPUTE))

            with pytest.raises(ValueError, match="Tile type mismatch"):
                dev.resolve_tile(worker.tile)


def test_duplicate_compute_tile_error():
    """Same compute tile cannot be allocated twice."""

    def kernel1():
        pass

    def kernel2():
        pass

    dev = NPU2()

    with mlir_mod_ctx() as ctx:

        @aie_device(AIEDevice.npu2)
        def device_body():
            # Two workers with same coordinates
            worker1 = Worker(kernel1, placement=Tile(0, 2))
            worker2 = Worker(kernel2, placement=Tile(0, 2))

            # First worker succeeds
            dev.resolve_tile(worker1.tile)

            # Second worker should fail - duplicate compute tile
            with pytest.raises(ValueError, match="already allocated"):
                dev.resolve_tile(worker2.tile)


def test_invalid_tile_type_string():
    """Invalid tile_type string should be rejected."""

    with pytest.raises(ValueError, match="Invalid tile_type"):
        Tile(tile_type="invalid")

    with pytest.raises(ValueError, match="Invalid tile_type"):
        Tile(tile_type="core")  # Should be "compute" not "core"


def test_objectfifo_link_rejects_shim():
    """ObjectFifoLink should reject shim tile type (only memory or compute allowed)."""

    n_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    of_in = ObjectFifo(n_ty, name="in")

    # Should reject shim type for forward()
    with pytest.raises(
        ValueError, match="ObjectFifoLink requires Tile.MEMORY or Tile.COMPUTE"
    ):
        of_out = of_in.cons().forward(placement=Tile(tile_type=Tile.SHIM))

    # Should accept memory (default)
    of_out = of_in.cons().forward()  # OK - defaults to MEMORY

    # Should accept compute (special case)
    of_out2 = of_in.cons().forward(placement=Tile(0, 2))  # OK if (0,2) is compute


if __name__ == "__main__":
    test_worker_rejects_wrong_tile_type()
    test_tile_type_coordinate_mismatch()
    test_duplicate_compute_tile_error()
    test_invalid_tile_type_string()
    test_objectfifo_link_rejects_shim()
