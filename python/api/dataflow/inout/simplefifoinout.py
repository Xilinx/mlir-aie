"""
TODO: 
* docs
* validation...?
"""

from .... import ir
from ....dialects.aiex import runtime_sequence, npu_dma_memcpy_nd, npu_sync, T
from .inout import InOutProgram
from ...phys.tile import MyTile
from ..objectfifo import ObjectFifoHandle


class SimpleFifoInOutProgram(InOutProgram):
    def __init__(
        self,
        fifo_in: ObjectFifoHandle,
        bytes_in: int,
        fifo_out: ObjectFifoHandle,
        bytes_out: int,
    ):
        # assert bytes_in % np.prod(fifo_in.__memref_type[0]) == 0
        # assert bytes_out % np.prod(fifo_out.__memref_type[0]) == 0
        self.fifo_in = fifo_in
        self.fifo_out = fifo_out
        self.bytes_in = bytes_in
        self.bytes_out = bytes_out
        self.tile = MyTile(0, 0)  # TODO: how to set default here?
        fifo_in.set_endpoint(self)
        fifo_out.set_endpoint(self)

    def get_tile(self) -> MyTile:
        assert self.tile != None
        return self.tile.op

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        tensor_in_ty = T.memref(self.bytes_in, T.ui8())
        tensor_out_ty = T.memref(self.bytes_out, T.ui8())

        @runtime_sequence(tensor_in_ty, tensor_out_ty)
        def sequence(inTensor, outTensor):
            npu_dma_memcpy_nd(
                metadata=self.fifo_out.name,
                bd_id=0,
                mem=inTensor,
                sizes=[1, 1, 1, self.bytes_out],
            )
            npu_dma_memcpy_nd(
                metadata=self.fifo_in.name,
                bd_id=1,
                mem=outTensor,
                sizes=[1, 1, 1, self.bytes_in],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)
