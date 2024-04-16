from functools import cached_property, reduce
from typing import Tuple

import numpy as np

from ....ir import DenseElementsAttr, ShapedType, Type

S = ShapedType.get_dynamic_size()


# mixin that requires `is_constant`
class ShapedValue:
    @cached_property
    def literal_value(self) -> np.ndarray:
        if not self.is_constant:
            raise ValueError("Can't build literal from non-constant value")
        return np.array(DenseElementsAttr(self.owner.opview.value), copy=False)

    @cached_property
    def _shaped_type(self) -> ShapedType:
        return ShapedType(self.type)

    def has_static_shape(self) -> bool:
        return self._shaped_type.has_static_shape

    def has_rank(self) -> bool:
        return self._shaped_type.has_rank

    @cached_property
    def rank(self) -> int:
        return self._shaped_type.rank

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shaped_type.shape)

    @cached_property
    def n_elements(self) -> int:
        assert self.has_static_shape()
        return reduce(lambda acc, v: acc * v, self._shaped_type.shape, 1)

    @cached_property
    def dtype(self) -> Type:
        return self._shaped_type.element_type
