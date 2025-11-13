import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.dialects.ext.func import func

# RUN: %python %s | FileCheck %s


def custom_loop_type(loop_dtype):
    # Define tensor types
    vector_size = 1024
    loop_iter = 4
    line_size = vector_size // loop_iter
    line_type = np.ndarray[(line_size,), np.dtype[np.uint8]]
    vector_type = np.ndarray[(vector_size,), np.dtype[np.uint8]]

    # Dataflow with ObjectFifos
    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    # A python function which will be treated as a callable function on the AIE
    # e.g., a kernel written in python
    @func
    def passthrough_fn(
        input: line_type, output: line_type, line_width: np.int32, iter_num: loop_dtype
    ):
        for i in range_(line_width):
            output[i] = input[i]

    # The task for the core to perform (the core entry point, if you will)
    def core_fn(of_in, of_out, passthrough_fn):
        for i in range_(loop_dtype(loop_iter)):
            elemOut = of_out.acquire(1)
            elemIn = of_in.acquire(1)
            passthrough_fn(elemIn, elemOut, line_size, loop_iter)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task
    my_worker = Worker(core_fn, [of_in.cons(), of_out.prod(), passthrough_fn])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(NPU2(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


# CHECK-LABEL: TEST: range_with_int32
# CHECK: scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
def range_with_int32():
    print("range_with_int32")
    print(custom_loop_type(np.int32))


range_with_int32()


# CHECK-LABEL: TEST: range_with_int64
# CHECK: scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
def range_with_int64():
    print("range_with_int64")
    print(custom_loop_type(np.int64))


range_with_int64()
