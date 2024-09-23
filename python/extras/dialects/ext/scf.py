from typing import Optional, Sequence

from ....ir import InsertionPoint, Value
from ....dialects.linalg.opdsl.lang.emitter import _is_index_type
from ....dialects.scf import ForOp, yield_

from .arith import constant, index_cast


def _for(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
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
    """
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            p = constant(p, index=True)
        if not _is_index_type(p.type):
            p = index_cast(p)
        params[i] = p

    start, stop, step = params

    for_op = ForOp(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = for_op.induction_variable
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
