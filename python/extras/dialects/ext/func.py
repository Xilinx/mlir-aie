import numpy as np
from functools import update_wrapper
import sys
from typing import get_args, get_origin

from ...meta import op_region_builder
from ...util import (
    get_arg_types,
    get_user_code_loc,
    make_maybe_no_args_decorator,
    NpuDType,
    try_convert_np_type_to_mlir_type,
)
from ....dialects._ods_common import get_op_result_or_op_results
from ....dialects.func import *
from ....ir import (
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    OpView,
    Operation,
    Type,
    TypeAttr,
    Value,
)
from .arith import Scalar


def call(
    callee_or_results: FuncOp | List[Type] | str,
    arguments_or_callee: (
        List[NpuDType | type[np.ndarray] | Value] | FlatSymbolRefAttr | str
    ),
    arguments: List[NpuDType | type[np.ndarray] | Value] | None = None,
    *,
    call_op_ctor=CallOp.__base__,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(callee_or_results, FuncOp.__base__):
        if not isinstance(arguments_or_callee, (list, tuple)):
            raise ValueError(
                "when constructing a call to a function, expected "
                + "the second argument to be a list of call arguments, "
                + f"got {type(arguments_or_callee)}"
            )
        if arguments is not None:
            raise ValueError(
                "unexpected third argument when constructing a call" + "to a function"
            )
        args = []
        for i, a in enumerate(arguments_or_callee):
            if isinstance(a, (int, float)):
                # Get the type to convert the python value to based on the expected input to the function
                # TODO: should check if it's safe to do this? What is int value is outside range?
                args.append(
                    Scalar(a, dtype=callee_or_results.function_type.value.inputs[i])
                )
            else:
                args.append(a)
        if not all(isinstance(a, (Value, Operation, OpView)) for a in args):
            raise ValueError(f"{args} must all be Value, Operation, or OpView")

        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results.function_type.value.results,
                FlatSymbolRefAttr.get(callee_or_results.sym_name.value),
                args,
                loc=loc,
                ip=ip,
            )
        )

    if isinstance(arguments_or_callee, list):
        raise ValueError(
            "when constructing a call to a function by name, "
            + "expected the second argument to be a string or a "
            + f"FlatSymbolRefAttr, got {type(arguments_or_callee)}"
        )

    if isinstance(arguments_or_callee, FlatSymbolRefAttr):
        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results, arguments_or_callee, arguments, loc=loc, ip=ip
            )
        )
    elif isinstance(arguments_or_callee, str):
        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results,
                FlatSymbolRefAttr.get(arguments_or_callee),
                arguments,
                loc=loc,
                ip=ip,
            )
        )
    else:
        raise ValueError(f"unexpected type {callee_or_results=}")


def isalambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def prep_func_types(sig, return_types):
    assert not (
        not sig.return_annotation is inspect.Signature.empty and len(return_types) > 0
    ), f"func can use return annotation or explicit return_types but not both"
    return_types = (
        sig.return_annotation
        if not sig.return_annotation is inspect.Signature.empty
        else return_types
    )
    if not isinstance(return_types, (tuple, list)):
        return_types = [return_types]
    return_types = list(return_types)
    assert all(
        isinstance(r, Type) for r in return_types
    ), f"all return types must be mlir types {return_types=}"

    input_types = [
        p.annotation
        for p in sig.parameters.values()
        if not p.annotation is inspect.Signature.empty
    ]
    # convert ndarray types to memref types
    assert all(
        isinstance(r, (str, Type))
        or isalambda(r)
        or get_origin(r) == np.ndarray
        or r in get_args(NpuDType)
        for r in input_types
    ), f"all input types must be mlir types or ndarrays (tensors) or np dtypes {input_types=}"
    user_loc = get_user_code_loc()
    # If ir.Context is none (like for deferred func emit)
    if user_loc is None:
        user_locs = None
    else:
        user_locs = [user_loc] * len(sig.parameters)
    return input_types, return_types, user_locs


class FuncBase:
    def __init__(
        self,
        body_builder,
        func_op_ctor,
        return_op_ctor,
        call_op_ctor,
        return_types=None,
        sym_visibility=None,
        arg_attrs=None,
        res_attrs=None,
        func_attrs=None,
        loc=None,
        ip=None,
        qualname=None,
    ):
        assert inspect.isfunction(body_builder), body_builder
        assert inspect.isclass(func_op_ctor), func_op_ctor
        assert inspect.isclass(return_op_ctor), return_op_ctor
        assert inspect.isclass(call_op_ctor), call_op_ctor

        self.body_builder = body_builder
        self.func_name = self.body_builder.__name__
        self.func_op_ctor = func_op_ctor
        self.return_op_ctor = return_op_ctor
        self.call_op_ctor = call_op_ctor
        self.arg_attrs = arg_attrs
        self.res_attrs = res_attrs
        self.loc = loc
        self.ip = ip
        self._func_op = None
        # in case this function lives inside a class
        self.qualname = qualname

        self.sym_visibility = sym_visibility
        self.func_attrs = func_attrs
        if self.func_attrs is None:
            self.func_attrs = {}

        if return_types is None:
            return_types = []
        sig = inspect.signature(self.body_builder)
        self.input_types, self.return_types, self.arg_locs = prep_func_types(
            sig, return_types
        )

        if self._is_decl():
            assert len(self.input_types) == len(
                sig.parameters
            ), f"func decl needs all input types annotated"
            self.sym_visibility = "private"
            self.emit()

    def _is_decl(self):
        # magic constant found from looking at the code for an empty fn
        if sys.version_info.minor == 12:
            return self.body_builder.__code__.co_code == b"\x97\x00y\x00"
        elif sys.version_info.minor == 11:
            return self.body_builder.__code__.co_code == b"\x97\x00d\x00S\x00"
        elif sys.version_info.minor in {8, 9, 10}:
            return self.body_builder.__code__.co_code == b"d\x00S\x00"
        else:
            raise NotImplementedError(f"{sys.version_info.minor} not supported.")

    def __str__(self):
        return str(f"{self.__class__} {self.__dict__}")

    def emit(self, *call_args, decl=False, force=False) -> FuncOp:
        if self._func_op is None or decl or force:
            input_types = self.input_types[:]
            for i, v in enumerate(input_types):
                if isinstance(v, str):
                    input_types[i] = Type(eval(v, self.body_builder.__globals__))
                elif isalambda(v):
                    input_types[i] = v()
                else:
                    new_type = try_convert_np_type_to_mlir_type(v)
                    if new_type != v:
                        input_types[i] = new_type
            if len(call_args) != 0:
                assert len(call_args) == len(input_types)

            function_type = TypeAttr.get(
                FunctionType.get(
                    inputs=input_types,
                    results=self.return_types,
                )
            )
            self._func_op = self.func_op_ctor(
                self.func_name,
                function_type,
                sym_visibility=self.sym_visibility,
                arg_attrs=self.arg_attrs,
                res_attrs=self.res_attrs,
                loc=self.loc,
                ip=self.ip or InsertionPoint.current,
            )
            for k, v in self.func_attrs.items():
                self._func_op.attributes[k] = v
            if self._is_decl() or decl:
                return self._func_op

            self._func_op.regions[0].blocks.append(*input_types, arg_locs=self.arg_locs)
            builder_wrapper = op_region_builder(
                self._func_op, self._func_op.regions[0], terminator=self.return_op_ctor
            )

            return_types = []

            def grab_results(*args):
                nonlocal return_types
                results = self.body_builder(*args)
                if isinstance(results, (tuple, list)):
                    return_types.extend(get_arg_types(results))
                elif results is not None:
                    return_types.extend(get_arg_types([results]))
                return results

            builder_wrapper(grab_results)

            function_type = FunctionType.get(inputs=input_types, results=return_types)
            self._func_op.attributes["function_type"] = TypeAttr.get(function_type)
        return self._func_op

    def __call__(self, *call_args):
        return call(self.emit(*call_args), call_args)


@make_maybe_no_args_decorator
def func(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    func_attrs=None,
    emit=False,
    loc=None,
    ip=None,
) -> FuncBase:
    if loc is None:
        loc = get_user_code_loc()
    func = FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp.__base__,
        return_op_ctor=ReturnOp,
        call_op_ctor=CallOp.__base__,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        func_attrs=func_attrs,
        loc=loc,
        ip=ip,
    )
    func = update_wrapper(func, f)
    if emit:
        func.emit()
    return func
