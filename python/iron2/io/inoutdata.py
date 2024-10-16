import numpy as np
from ...extras.dialects.ext.memref import MemRef


class InOutData:
    def __init__(self, inout_type: type[np.ndarray]):
        self.inout_type = inout_type
        self.__op = None

    @property
    def op(self) -> MemRef:
        assert self.__op != None
        return self.__op

    @op.setter
    def op(self, op: MemRef):
        assert self.__op == None
        self.__op = op
