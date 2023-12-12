import ctypes
import inspect
import sys
from functools import wraps
from pathlib import Path
from typing import Optional
from typing import Sequence

from ..dialects._ods_common import (
    get_op_result_or_op_results,
)
from ..ir import Context, Location, Type, InsertionPoint


def get_user_code_loc(user_base: Optional[Path] = None):
    if Context.current is None:
        return

    mlir_utils_root_path = Path(__file__).parent

    prev_frame = inspect.currentframe().f_back
    if user_base is None:
        user_base = Path(prev_frame.f_code.co_filename)

    while prev_frame.f_back and (
        Path(prev_frame.f_code.co_filename).is_relative_to(mlir_utils_root_path)
        or Path(prev_frame.f_code.co_filename).is_relative_to(sys.prefix)
        or Path(prev_frame.f_code.co_filename).is_relative_to(user_base)
    ):
        prev_frame = prev_frame.f_back
    frame_info = inspect.getframeinfo(prev_frame)
    if sys.version_info.minor >= 11:
        return Location.file(
            frame_info.filename, frame_info.lineno, frame_info.positions.col_offset
        )
    elif sys.version_info.minor == 10:
        return Location.file(frame_info.filename, frame_info.lineno, col=0)
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")


# builds the decorator
def make_maybe_no_args_decorator(decorator):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(decorator)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return decorator(args[0])
        else:
            # decorator arguments
            return lambda realf: decorator(realf, *args, **kwargs)

    return new_dec


def op_region_builder(op, op_region, terminator=None):
    def builder_wrapper(body_builder):
        # add a block with block args having types ...
        if len(op_region.blocks) == 0:
            sig = inspect.signature(body_builder)
            types = [p.annotation for p in sig.parameters.values()]
            if not (
                len(types) == len(sig.parameters)
                and all(isinstance(t, Type) for t in types)
            ):
                raise ValueError(
                    f"for {body_builder=} either missing a type annotation or type annotation isn't a mlir type: {sig}"
                )

            arg_locs = list(filter(None, [get_user_code_loc()] * len(sig.parameters)))
            if len(arg_locs) == 0:
                arg_locs = None
            op_region.blocks.append(*types, arg_locs=arg_locs)

        with InsertionPoint(op_region.blocks[0]):
            results = body_builder(*list(op_region.blocks[0].arguments))

        with InsertionPoint(list(op_region.blocks)[-1]):
            if terminator is not None:
                res = []
                if isinstance(results, (tuple, list)):
                    res.extend(results)
                elif results is not None:
                    res.append(results)
                terminator(res)

        return get_op_result_or_op_results(op)

    return builder_wrapper


def region_adder(terminator=None):
    def wrapper(op_region_adder):
        def region_adder_decorator(op, *args, **kwargs):
            region = op_region_adder(op, *args, **kwargs)

            return op_region_builder(op, region, terminator)

        return region_adder_decorator

    return wrapper


def region_op(op_constructor, terminator=None):
    # the decorator itself
    def op_decorator(*args, **kwargs):
        op = op_constructor(*args, **kwargs)
        op_region = op.regions[0]

        return op_region_builder(op, op_region, terminator)

    # this is like make_maybe_no_args_decorator but a little different because the decorators here
    # are already wrapped (or something like that)
    @wraps(op_decorator)
    def maybe_no_args(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return op_decorator()(args[0])
        else:
            return op_decorator(*args, **kwargs)

    return maybe_no_args


def _update_caller_vars(previous_frame, args: Sequence, replacements: Sequence):
    """Update caller vars passed as args.

    This function uses CPython API  to update the values
    of the caller's args (not the caller of this function but the caller of caller of this function).
    It does this by searching for a match in the caller's f_locals based on identity (A is A) and then
    updating all corresponding values in the f_locals dict. Finally, it uses PyFrame_LocalsToFast to signal
    to the CPython runtime that an update has been made to f_locals.

    Args:
      previous_frame: The frame in which vars will be updated.
      args: The args to the callee.
      replacements: The values that should replace the values of the vars in the caller.
    """

    if len(args) != len(replacements):
        raise ValueError(f"updates must be 1-1: {args=} {replacements=}")
    # find the name of the iter args in the previous frame
    var_names = [
        [
            var_name
            for var_name, var_val in previous_frame.f_locals.items()
            if var_val is arg
        ]
        for arg in args
    ]
    for i, var_names in enumerate(var_names):
        for var_name in var_names:
            previous_frame.f_locals[var_name] = replacements[i]
            # signal to update
            # for some reason you can only update one at a time?
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(previous_frame), ctypes.c_int(1)
            )


class ModuleMeta(type):
    def __new__(cls, name, bases, classdict, **kwargs):
        ip = classdict.pop("ip")
        loc = classdict.pop("loc")
        module_terminator = classdict.pop("module_terminator", None)
        new = super().__new__(cls, name, bases, classdict)
        if module_terminator is not None:
            module_terminator(loc=loc, ip=ip)
        for k, v in classdict.items():
            if callable(v):
                v.qualname = name
        ip.__exit__(None, None, None)
        return new
