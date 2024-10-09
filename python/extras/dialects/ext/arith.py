import operator
from abc import abstractmethod
from copy import deepcopy
from functools import cached_property, partialmethod
import numpy as np
from typing import Tuple

from ...util import get_user_code_loc, infer_mlir_type, mlir_type_to_np_dtype
from ...._mlir_libs._mlir import register_value_caster
from ....dialects import arith as arith_dialect, complex as complex_dialect
from ....dialects._arith_enum_gen import (
    _arith_cmpfpredicateattr,
    _arith_cmpipredicateattr,
)
from ....dialects._ods_common import get_op_result_or_op_results, get_op_result_or_value
from ....dialects.arith import *
from ....dialects.arith import _is_integer_like_type
from ....dialects.linalg.opdsl.lang.emitter import (
    _is_complex_type,
    _is_floating_point_type,
    _is_index_type,
)
from ....ir import (
    Attribute,
    BF16Type,
    ComplexType,
    Context,
    DenseElementsAttr,
    F16Type,
    F32Type,
    F64Type,
    FloatAttr,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    OpView,
    Operation,
    ShapedType,
    Type,
    Value,
    register_attribute_builder,
)


def constant(
    value: int | float | bool | np.ndarray,
    type: Type | None = None,
    index: bool | None = None,
    *,
    vector: bool | None = False,
    loc: Location | None = None,
    ip: InsertionPoint | None = None,
) -> Value:
    """Instantiate arith.constant with value `value`.

    Args:
      value: Python value that determines the value attribute of the
        arith.constant op.
      type: Optional MLIR type that type of the value attribute of the
        arith.constant op; if omitted the type of the value attribute
        will be inferred from the value.
      index: Whether the MLIR type should be an index type; if passed the
        type argument will be ignored.

    Returns:
      ir.OpView instance that corresponds to instantiated arith.constant op.
    """
    if loc is None:
        loc = get_user_code_loc()
    if index is not None and index:
        type = IndexType.get()
    if type is None:
        type = infer_mlir_type(value, vector=vector)

    assert type is not None

    if _is_complex_type(type):
        value = complex(value)
        return get_op_result_or_op_results(
            complex_dialect.ConstantOp(
                type,
                list(
                    map(
                        lambda x: FloatAttr.get(type.element_type, x),
                        [value.real, value.imag],
                    )
                ),
                loc=loc,
                ip=ip,
            )
        )

    if _is_floating_point_type(type) and not isinstance(value, np.ndarray):
        value = float(value)

    if ShapedType.isinstance(type) and isinstance(value, (int, float, bool)):
        ranked_tensor_type = ShapedType(type)
        value = np.full(
            ranked_tensor_type.shape,
            value,
            dtype=mlir_type_to_np_dtype(ranked_tensor_type.element_type),
        )

    if isinstance(value, np.ndarray):
        value = DenseElementsAttr.get(
            value,
            type=type,
        )

    return get_op_result_or_op_results(
        arith_dialect.ConstantOp(type, value, loc=loc, ip=ip)
    )


def index_cast(
    value: Value,
    *,
    to: Type = None,
    loc: Location = None,
    ip: InsertionPoint = None,
) -> Value:
    if loc is None:
        loc = get_user_code_loc()
    if to is None:
        to = IndexType.get()
    return get_op_result_or_op_results(
        arith_dialect.IndexCastOp(to, value, loc=loc, ip=ip)
    )


class ArithValueMeta(type(Value)):
    """Metaclass that orchestrates the Python object protocol
    (i.e., calling __new__ and __init__) for Indexing dialect extension values
    (created using `mlir_value_subclass`).

    The purpose/benefit of handling the `__new__` and `__init__` calls
    explicitly/manually is we can then wrap arbitrary Python objects; e.g.
    all three of the following wrappers are equivalent:

    ```
    s1 = Scalar(arith.ConstantOp(f64, 0.0).result)
    s2 = Scalar(arith.ConstantOp(f64, 0.0))
    s3 = Scalar(0.0)
    ```

    In general the Python object protocol for an object instance is determined
    by `__call__` of the object class's metaclass, thus here we overload
    `__call__` and branch on what we're wrapping there.

    Why not just overload __new__ and be done with it? Because then we can't
    choose what get's passed to __init__: by default (i.e., without overloading
    __call__ here) the same arguments are passed to both __new__ and __init__.

    Note, this class inherits from `type(Value)` (i.e., the metaclass of
    `ir.Value`) rather than `type` or `abc.ABCMeta` or something like this because
    the metaclass of a derived class must be a (non-strict) subclass of the
    metaclasses of all its bases and so all the extension classes
    (`ScalarValue`, `TensorValue`), which are derived classes of `ir.Value` must
    have metaclasses that inherit from the metaclass of `ir.Value`. Without this
    hierarchy Python will throw `TypeError: metaclass conflict`.
    """

    def __call__(cls, *args, **kwargs):
        """Orchestrate the Python object protocol for mlir
        values in order to handle wrapper arbitrary Python objects.

        Args:
          *args: Position arguments to the class constructor. Note, currently,
            only one positional arg is supported (so constructing something like a
            tuple type from element objects isn't supported).
          **kwargs: Keyword arguments to the class constructor. Note, currently,
            we only look for `dtype` (an `ir.Type`).

        Returns:
          A fully constructed and initialized instance of the class.
        """
        if len(args) != 1:
            raise ValueError("Only one non-kw arg supported.")
        arg = args[0]
        fold = None
        if isinstance(arg, (OpView, Operation, Value)):
            # wrap an already created Value (or op the produces a Value)
            if isinstance(arg, (Operation, OpView)):
                assert len(arg.results) == 1
            val = get_op_result_or_value(arg)
        elif isinstance(arg, (int, float, bool, np.ndarray)):
            # wrap a Python value, effectively a scalar or tensor literal
            dtype = kwargs.get("dtype")
            if dtype is not None and not isinstance(dtype, Type):
                raise ValueError(f"{dtype=} is expected to be an ir.Type.")
            fold = kwargs.get("fold")
            if fold is not None and not isinstance(fold, bool):
                raise ValueError(f"{fold=} is expected to be a bool.")
            loc = kwargs.get("loc")
            ip = kwargs.get("ip")
            # If we're wrapping a numpy array (effectively a tensor literal),
            # then we want to make sure no one else has access to that memory.
            # Otherwise, the array will get funneled down to DenseElementsAttr.get,
            # which by default (through the Python buffer protocol) does not copy;
            # see mlir/lib/Bindings/Python/IRAttributes.cpp#L556
            val = constant(deepcopy(arg), dtype, loc=loc, ip=ip)
        else:
            raise NotImplementedError(f"{cls.__name__} doesn't support wrapping {arg}.")

        # The mlir_value_subclass mechanism works through __new__
        # (see mlir/Bindings/Python/PybindAdaptors.h#L502)
        # So we have to pass the wrapped Value to the __new__ of the subclass
        cls_obj = cls.__new__(cls, val)
        # We also have to pass it to __init__ because that is required by
        # the Python object protocol; first an object is new'ed and then
        # it is init'ed. Note we pass arg_copy here in case a subclass wants to
        # inspect the literal.
        cls.__init__(cls_obj, val, fold=fold)
        return cls_obj


@register_attribute_builder("Arith_CmpIPredicateAttr", replace=True)
def _arith_CmpIPredicateAttr(predicate: str | Attribute, context: Context):
    predicates = {
        "eq": CmpIPredicate.eq,
        "ne": CmpIPredicate.ne,
        "slt": CmpIPredicate.slt,
        "sle": CmpIPredicate.sle,
        "sgt": CmpIPredicate.sgt,
        "sge": CmpIPredicate.sge,
        "ult": CmpIPredicate.ult,
        "ule": CmpIPredicate.ule,
        "ugt": CmpIPredicate.ugt,
        "uge": CmpIPredicate.uge,
    }
    if isinstance(predicate, Attribute):
        return predicate
    assert predicate in predicates, f"{predicate=} not in predicates"
    return _arith_cmpipredicateattr(predicates[predicate], context)


@register_attribute_builder("Arith_CmpFPredicateAttr", replace=True)
def _arith_CmpFPredicateAttr(predicate: str | Attribute, context: Context):
    predicates = {
        "false": CmpFPredicate.AlwaysFalse,
        # ordered comparison
        # An ordered comparison checks if neither operand is NaN.
        "oeq": CmpFPredicate.OEQ,
        "ogt": CmpFPredicate.OGT,
        "oge": CmpFPredicate.OGE,
        "olt": CmpFPredicate.OLT,
        "ole": CmpFPredicate.OLE,
        "one": CmpFPredicate.ONE,
        # no clue what this one is
        "ord": CmpFPredicate.ORD,
        # unordered comparison
        # Conversely, an unordered comparison checks if either operand is a NaN.
        "ueq": CmpFPredicate.UEQ,
        "ugt": CmpFPredicate.UGT,
        "uge": CmpFPredicate.UGE,
        "ult": CmpFPredicate.ULT,
        "ule": CmpFPredicate.ULE,
        "une": CmpFPredicate.UNE,
        # no clue what this one is
        "uno": CmpFPredicate.UNO,
        # return always true
        "true": CmpFPredicate.AlwaysTrue,
    }
    if isinstance(predicate, Attribute):
        return predicate
    assert predicate in predicates, f"{predicate=} not in predicates"
    return _arith_cmpfpredicateattr(predicates[predicate], context)


def _binary_op(
    lhs: "ArithValue",
    rhs: "ArithValue",
    op: str,
    predicate: str | None = None,
    signedness: str | None = None,
    *,
    loc: Location | None = None,
) -> "ArithValue":
    """Generic for handling infix binary operator dispatch.

    Args:
      lhs: E.g. Scalar or Tensor below.
      rhs: Scalar or Tensor with type matching self.
      op: Binary operator, currently only add, sub, mul
        supported.

    Returns:
      Result of binary operation. This will be a handle to an arith(add|sub|mul) op.
    """
    if loc is None:
        loc = get_user_code_loc()
    if (
        isinstance(rhs, Value)
        and lhs.type != rhs.type
        or isinstance(rhs, (float, int, bool, np.ndarray))
    ):
        lhs, rhs = lhs.coerce(rhs)
    assert lhs.type == rhs.type, f"{lhs=} {rhs=} must have the same type."

    assert op in {"add", "and", "or", "sub", "mul", "cmp", "truediv", "floordiv", "mod"}

    if op == "cmp":
        assert predicate is not None

    if lhs.fold() and lhs.fold():
        klass = lhs.__class__
        # if both operands are constants (results of an arith.constant op)
        # then both have a literal value (i.e. Python value).
        lhs, rhs = lhs.literal_value, rhs.literal_value
        # if we're folding constants (self._fold = True) then we just carry out
        # the corresponding operation on the literal values; e.g., operator.add.
        # note this is the same as op = operator.__dict__[op].
        if predicate is not None:
            op = predicate
        op = operator.attrgetter(op)(operator)
        return klass(op(lhs, rhs), fold=True)

    if op == "truediv":
        op = "div"
    if op == "mod":
        op = "rem"

    op = op.capitalize()
    if _is_floating_point_type(lhs.dtype):
        if op == "Floordiv":
            raise ValueError(f"floordiv not supported for {lhs=}")
        op += "F"
    elif _is_integer_like_type(lhs.dtype):
        # TODO(max): this needs to all be regularized
        if "div" in op.lower() or "rem" in op.lower():
            if not lhs.dtype.is_signless:
                raise ValueError(f"{op.lower()}i not supported for {lhs=}")
            if op == "Floordiv":
                op = "FloorDiv"
            op += "S"
        op += "I"
    else:
        raise NotImplementedError(f"Unsupported '{op}' operands: {lhs}, {rhs}")

    op = getattr(arith_dialect, f"{op}Op")

    if predicate is not None:
        if _is_floating_point_type(lhs.dtype):
            # ordered comparison - see above
            predicate = "o" + predicate
        elif _is_integer_like_type(lhs.dtype):
            # eq, ne signs don't matter
            if predicate not in {"eq", "ne"}:
                if signedness is not None:
                    predicate = signedness + predicate
                else:
                    if lhs.dtype.is_signed:
                        predicate = "s" + predicate
                    else:
                        predicate = "u" + predicate
        return lhs.__class__(op(predicate, lhs, rhs, loc=loc), dtype=lhs.dtype)
    else:
        return lhs.__class__(op(lhs, rhs, loc=loc), dtype=lhs.dtype)


# TODO(max): these could be generic in the dtype
# TODO(max): hit .verify() before constructing (maybe)
class ArithValue(Value, metaclass=ArithValueMeta):
    """Class for functionality shared by Value subclasses that support
    arithmetic operations.

    Note, since we bind the ArithValueMeta here, it is here that the __new__ and
    __init__ must be defined. To be precise, the callchain, starting from
    ArithValueMeta is:

    ArithValueMeta.__call__ -> mlir_value_subclass.__new__ ->
                          (mlir_value_subclass.__init__ == ArithValue.__init__) ->
                          Value.__init__
    """

    def __init__(self, val, *, fold: bool | None = None):
        self._fold = fold if fold is not None else False
        super().__init__(val)

    def is_constant(self) -> bool:
        return isinstance(self.owner, Operation) and isinstance(
            self.owner.opview, arith_dialect.ConstantOp
        )

    @property
    @abstractmethod
    def literal_value(self):
        pass

    @abstractmethod
    def coerce(self, other) -> Tuple["ArithValue", "ArithValue"]:
        pass

    def fold(self) -> bool:
        return self.is_constant() and self._fold

    def __str__(self):
        return f"{self.__class__.__name__}({self.get_name()}, {self.type})"

    def __repr__(self):
        return str(Value(self)).replace("Value", self.__class__.__name__)

    # partialmethod differs from partial in that it also binds the object instance
    # to the first arg (i.e., self)
    __add__ = partialmethod(_binary_op, op="add")
    __sub__ = partialmethod(_binary_op, op="sub")
    __mul__ = partialmethod(_binary_op, op="mul")
    __truediv__ = partialmethod(_binary_op, op="truediv")
    __floordiv__ = partialmethod(_binary_op, op="floordiv")
    __mod__ = partialmethod(_binary_op, op="mod")

    __radd__ = partialmethod(_binary_op, op="add")
    __rsub__ = partialmethod(_binary_op, op="sub")
    __rmul__ = partialmethod(_binary_op, op="mul")

    __and__ = partialmethod(_binary_op, op="and")
    __or__ = partialmethod(_binary_op, op="or")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other, dtype=self.type)
            except NotImplementedError as e:
                assert "doesn't support wrapping" in str(e)
                return False
        if self is other:
            return True
        return _binary_op(self, other, op="cmp", predicate="eq")

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other, dtype=self.type)
            except NotImplementedError as e:
                assert "doesn't support wrapping" in str(e)
                return True
        if self is other:
            return False
        return _binary_op(self, other, op="cmp", predicate="ne")

    __le__ = partialmethod(_binary_op, op="cmp", predicate="le")
    __lt__ = partialmethod(_binary_op, op="cmp", predicate="lt")
    __ge__ = partialmethod(_binary_op, op="cmp", predicate="ge")
    __gt__ = partialmethod(_binary_op, op="cmp", predicate="gt")

    def _eq(self, other):
        return Value(self) == Value(other)

    def _ne(self, other):
        return Value(self) != Value(other)


class Scalar(ArithValue):
    """Value subclass ScalarValue that adds convenience methods
    for getting dtype and (possibly) the stored literal value.

    Note, order matters in the superclasses above; ArithValue is first so that
    e.g. __init__, and __str__ from ArithValue are used instead of
    from ScalarValue.
    """

    @cached_property
    def dtype(self) -> Type:
        return self.type

    @cached_property
    def literal_value(self) -> int | float | bool:
        if not self.is_constant():
            raise ValueError("Can't build literal from non-constant Scalar")
        return self.owner.opview.literal_value

    def __int__(self):
        return int(self.literal_value)

    def __float__(self):
        return float(self.literal_value)

    def coerce(self, other) -> Tuple["Scalar", "Scalar"]:
        if isinstance(other, (int, float, bool)):
            other = Scalar(other, dtype=self.dtype)
        elif isinstance(other, Scalar) and _is_index_type(self.type):
            other = index_cast(other)
        elif isinstance(other, Scalar) and _is_index_type(other.type):
            other = index_cast(other, to=self.type)
        else:
            raise ValueError(f"can't coerce {other=} to {self=}")
        return self, other


for t in [BF16Type, F16Type, F32Type, F64Type, IndexType, IntegerType, ComplexType]:
    register_value_caster(t.static_typeid)(Scalar)
