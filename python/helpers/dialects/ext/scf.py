import numpy as np
from typing import get_args, Sequence

from ....ir import InsertionPoint, Value
from ....dialects.linalg.opdsl.lang.emitter import _is_index_type
from ....dialects.scf import ForOp, yield_
from ....extras.dialects.ext.arith import constant, index_cast, Scalar
from ...util import NpuDType, np_dtype_to_mlir_type


def _for(
    start,
    stop=None,
    step=None,
    iter_args: Sequence[Value] | None = None,
    insert_yield: bool = True,
    *,
    loc=None,
    ip=None,
):
    """
    This is nearly identical to the convenience wrapper in scf, but with the added insert_yield parameter.
    The insert_yield parameter defaults to True; if left as True, the user no longer needs to manually insert
    yield operations (```yield_([])```). If the user wishes to specify yield directly (such as if there is
    a return value from the loop body), insert_yield should be set to False.

    This also includes some convenience code that casts the return value to match the given
    values of the parameters (start, stop, step)
    """
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    params = [start, stop, step]
    param_dtypes = set()
    for i, p in enumerate(params):
        if isinstance(p, int) or type(p) in get_args(NpuDType):
            if not isinstance(p, int):
                param_dtypes.add(np_dtype_to_mlir_type(type(p)))

                # TODO: these should be a better way to convert than converting to
                # np values to python values but I don't know it.
                if isinstance(type(p), np.floating):
                    p = float(p)
                else:
                    p = int(p)
            p = constant(p, index=True)
        if not _is_index_type(p.type):
            p = index_cast(p)
        params[i] = p

    dtype = None
    start, stop, step = params
    if len(param_dtypes) > 1:
        raise ValueError(
            f"For-loop (_for) params have mismatched data types: {param_dtypes}"
        )
    if len(param_dtypes) == 1:
        dtype = param_dtypes.pop()

    for_op = ForOp(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = for_op.induction_variable
    if dtype:
        iv.set_type(dtype)
    iter_args = tuple(for_op.inner_iter_args)
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args, for_op.results
        elif len(iter_args) == 1:
            yield iv, iter_args[0], for_op.results[0]
        else:
            yield iv
        if insert_yield:
            yield_([])
