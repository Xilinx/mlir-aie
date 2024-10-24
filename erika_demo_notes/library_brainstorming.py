

import numpy as np

from aie.api.ml_lib import EltwiseAdd, EltwiseMul # Does not exist yet

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.placers import SequentialPlacer
from aie.api.program import Program
from aie.api.phys.device import NPU1Col4
from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D

M = ...
N = ...
tensor_ty = np.ndarray[(M,N), np.dtype[np.uint8]]

of_a = ObjectFifo(2, tensor_ty, "inA")
of_b = ObjectFifo(2, tensor_ty, "inB")

# Does not exist yet start
add_workers, of_add_out = EltwiseAdd(of_a.second, of_b.second, n_cores=4, ...)
mul_workers, of_c = EltwiseMul(of_add_out.second, of_add_out.second, n_cores=4, ...)
# Does not exist yet end

io = IOCoordinator()
with io.build_sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, b_in, c_out):
    tiler = TensorTiler2D(M, N)
    for t in io.tile_loop(tiler.tile_iter()):
        io.fill(of_a.first, t, a_in)
        io.fill(of_b.first, t, b_in)
        io.drain(of_c.second, t, c_out)

my_program = Program(NPU1Col4(), io, workers=add_workers + mul_workers)
my_program.resolve_program(SequentialPlacer())

# Notes:
# - May need some notion of worker shape
# - Do you just rely on documentation for out data format/location 
#   or do you have something like a spensor object?


##################################################################################
# Alternate version with more placement control
import numpy as np

from aie.api.ml_lib import EltwiseAdd, EltwiseMul

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.placers import SequentialPlacer
from aie.api.program import Program
from aie.api.phys.device import NPU1Col4
from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D

dev = NPU1Col4()

M = ...
N = ...
tensor_ty = np.ndarray[(M,N), np.dtype[np.uint8]]

of_a = ObjectFifo(2, tensor_ty, "inA")
of_b = ObjectFifo(2, tensor_ty, "inB")

add_workers, of_add_out = EltwiseAdd(of_a.second, of_b.second, n_cores=4, placement_resources=dev.tiles[0])
mul_workers, of_c = EltwiseMul(of_add_out.second, of_add_out.second, n_cores=4, placement_resources=dev.tiles[1])

io = IOCoordinator()
with io.build_sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, b_in, c_out):
    tiler = TensorTiler2D(M, N)
    for t in io.tile_loop(tiler.tile_iter()):
        io.fill(of_a.first, t, a_in, placement=dev.tiles[0].shim)
        io.fill(of_b.first, t, b_in, placement=dev.tiles[0].shim)
        io.drain(of_c.second, t, c_out, placement=dev.tiles[1].shim)

my_program = Program(dev, io, workers=add_workers + mul_workers)
my_program.resolve_program(SequentialPlacer())