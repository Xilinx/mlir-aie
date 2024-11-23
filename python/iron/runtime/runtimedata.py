import numpy as np
from ...extras.dialects.ext.memref import MemRef


class RuntimeData:
    def __init__(self, dtype: type[np.ndarray]):
        self.dtype = dtype
        self._op = None

    @property
    def op(self) -> MemRef:
        assert self._op != None
        return self._op

    @op.setter
    def op(self, op: MemRef):
        assert self._op == None
        self._op = op
