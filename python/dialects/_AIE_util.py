import inspect
from functools import wraps

from ..ir import *
from ..dialects import arith
from ..dialects import memref
from ..dialects.func import *
from ..dialects.scf import *


# Create a signless arith constant of given width (default is i32).
class integerConstant(arith.ConstantOp):
    """Specialize ConstantOp class constructor to take python integers"""
    def __init__(self, val, width = 32):
        if isinstance(width, int):
            int_ty = IntegerType.get_signless(width)
        else:
            int_ty = width
        intAttr = IntegerAttr.get(int_ty, val)
        super().__init__(result=int_ty, value=intAttr)


# Create an index arith constant.
class indexConstant(arith.ConstantOp):
    """Specialize ConstantOp class constructor to take python integers"""
    def __init__(self, val):
        idx_ty = IndexType.get()
        idxAttr = IntegerAttr.get(idx_ty, val)
        super().__init__(result=idx_ty, value=idxAttr)


class AddI(arith.AddIOp):
    """Specialize AddIOp class constructor to take python integers"""
    def __init__(self, lhs, rhs):
        if isinstance(lhs, int):
            intLhs = integerConstant(lhs)
        else:
            intLhs = lhs
        intRhs = integerConstant(rhs)
        super().__init__(lhs=intLhs, rhs=intRhs)


class For(ForOp):
    """Specialize ForOp class constructor to take python integers"""
    def __init__(self, lowerBound, upperBound, step):
        idxLowerBound = indexConstant(lowerBound) 
        idxUpperBound = indexConstant(upperBound)
        idxStep = indexConstant(step)
        super().__init__(lower_bound=idxLowerBound, upper_bound=idxUpperBound, step=idxStep)


# Wrapper for func FuncOp with "private" visibility.
class privateFunc(FuncOp):
    """Specialize FuncOp class constructor to take python integers"""
    def __init__(self, name, inputs, outputs = [], visibility = "private"):
        super().__init__(
            name=name, 
            type=FunctionType.get(inputs, outputs), 
            visibility=visibility
        )


# Wrapper for func FuncOp with "private" visibility.
class publicFunc(FuncOp):
    """Specialize FuncOp class constructor to take python integers"""
    def __init__(self, name, callbackFunc, inputs, outputs = [], visibility = "public"):
        super().__init__(
            name=name, 
            type=FunctionType.get(inputs, outputs), 
            visibility=visibility, 
            body_builder=callbackFunc
        )


# Wrapper for func CallOp.
class Call(CallOp):
    """Specialize CallOp class constructor to take python integers"""
    def __init__(self, calleeOrResults, inputs = [], input_types = []):
        attrInputs = []
        for i in inputs:
            if isinstance(i, int):
                attrInputs.append(integerConstant(i))
            else:
                attrInputs.append(i)
        if isinstance(calleeOrResults, FuncOp):
            super().__init__(calleeOrResults=calleeOrResults, argumentsOrCallee=attrInputs)
        else:
            super().__init__(
                calleeOrResults=input_types, 
                argumentsOrCallee=FlatSymbolRefAttr.get(calleeOrResults), 
                arguments=attrInputs
            )


class Load(memref.LoadOp):
    """Specialize LoadOp class constructor to take python integers"""
    def __init__(self, mem, indices):
        valueIndices = []
        if isinstance(indices, list):
            for i in indices:
                valueIndices.append(indexConstant(i))
        else:
            valueIndices.append(indexConstant(indices))
        super().__init__(memref=mem, indices=valueIndices)


class Store(memref.StoreOp):
    """Specialize StoreOp class constructor to take python integers"""
    def __init__(self, val, mem, indices):
        if isinstance(val, int):
            intVal = integerConstant(val)
        else:
            intVal = val
        valueIndices = []
        if isinstance(indices, list):
            for i in indices:
                valueIndices.append(indexConstant(i))
        else:
            valueIndices.append(indexConstant(indices))
        super().__init__(value=intVal, memref=mem, indices=valueIndices)


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
            op_region.blocks.append(*types)
        with InsertionPoint(op_region.blocks[0]):
            results = body_builder()
            if terminator is not None:
                res = []
                if isinstance(results, (tuple, list)):
                    res.extend(results)
                elif results is not None:
                    res.append(results)
                terminator(res)

        return op

    return builder_wrapper


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
