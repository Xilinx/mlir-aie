import numpy as np
from ...extras.dialects.ext.memref import MemRef


class InOutData:
    def __init__(self, inout_type: type[np.ndarray]):
        self.inout_type = inout_type
        self._op = None

    @property
    def op(self) -> MemRef:
        assert self._op != None
        return self._op

    @op.setter
    def op(self, op: MemRef):
        assert self._op == None
        self._op = op
