from typing import Sequence

from ....ir import InsertionPoint, Value
from ....dialects.linalg.opdsl.lang.emitter import _is_index_type
from ....dialects.scf import IfOp, ForOp, yield_
from ....extras.dialects.ext.arith import constant, index_cast
from ....extras.util import get_user_code_loc
from contextlib import contextmanager
from ....extras import types as T


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
            p = index_cast(p, to=T.index())
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


@contextmanager
def if_(cond, hasElse=True, insert_yield=True, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if_op = IfOp(cond, hasElse=hasElse, loc=loc, ip=ip)
    with InsertionPoint(if_op.thenRegion.blocks[0]):
        yield if_op
        if insert_yield:
            yield_([])
    if hasElse:
        with InsertionPoint(if_op.elseRegion.blocks[0]):
            if insert_yield:
                yield_([])


# This assumes that the parent if_ context manager passes down the else region.
@contextmanager
def else_(if_op, insert_yield=True):
    with InsertionPoint.at_block_terminator(if_op.elseRegion.blocks[0]):
        yield
