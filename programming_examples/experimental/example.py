import numpy as np
import aie.iron.experimental as aie

MATRIX_DIMS = (8, 16)
TILE_DIMS = (2, 4)
MATRIX_DTYPE = np.int32
NUM_WORKERS = 2

A = aie.asarray(np.full(fill_value=1, shape=MATRIX_DIMS, dtype=MATRIX_DTYPE))
B = aie.array(MATRIX_DIMS, MATRIX_DTYPE)


def task_fn(a, b):
    dim0, dim1 = a.shape
    for i in aie.range(dim0):
        for j in aie.range(dim1):
            b[i, j] = a[i, j] + 1


task_runner = aie.task_runner(task_fn, [(A, TILE_DIMS)], [(B, TILE_DIMS)], NUM_WORKERS)
task_runner.run()

npB = B.asnumpy()
npA = A.asnumpy()
if (npB == npA + 1).all():
    print("PASS!")
else:
    print(f"Failed: {np.B}")