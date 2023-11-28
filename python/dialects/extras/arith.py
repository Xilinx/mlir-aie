import numpy as np
from typing import Union, Optional

from ..arith import *
from .. import complex
from ..linalg.opdsl.lang.emitter import (
    _is_floating_point_type,
    _is_complex_type,
)
from ...ir import (
    DenseElementsAttr,
    FloatAttr,
    IndexType,
    InsertionPoint,
    Location,
    RankedTensorType,
    Type,
    Value,
)
from ...util import infer_mlir_type


def constant(
    value: Union[int, float, bool, np.ndarray],
    type: Optional[Type] = None,
    index: Optional[bool] = None,
    *,
    loc: Location = None,
    ip: InsertionPoint = None,
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
    if index is not None and index:
        type = IndexType.get()
    if type is None:
        type = infer_mlir_type(value)

    assert type is not None

    if _is_complex_type(type):
        value = complex(value)
        return complex.ConstantOp(
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

    if _is_floating_point_type(type) and not isinstance(value, np.ndarray):
        value = float(value)

    if RankedTensorType.isinstance(type) and isinstance(value, (int, float, bool)):
        ranked_tensor_type = RankedTensorType(type)
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

    return ConstantOp(type, value, loc=loc, ip=ip)
