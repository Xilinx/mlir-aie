"""
TODO: 
* docs
* validation...?
"""

import numpy as np
from typing import Optional

from .... import ir
from ....dialects.aiex import runtime_sequence, npu_dma_memcpy_nd, npu_sync, T
from .inout import InOutProgram
from ...phys.tile import MyTile
from ..objectfifo import ObjectFifoHandle
from ...tensor import MyTensorType


class SimpleFifoInOutProgram(InOutProgram):
    def __init__(
        self,
        fifo_in: ObjectFifoHandle,
        bytes_in: int,
        fifo_out: ObjectFifoHandle,
        bytes_out: int,
        in_sizes: Optional[list[int]] = None,
        in_strides: Optional[list[int]] = None,
        out_sizes: Optional[list[int]] = None,
        out_strides: Optional[list[int]] = None,
        dtype: np.generic = np.uint8,
    ):
        assert bytes_in % np.prod(fifo_in.obj_type.shape) == 0
        assert bytes_out % np.prod(fifo_in.obj_type.shape) == 0
        assert bytes_in > 0
        assert bytes_out > 0

        self.fifo_in = fifo_in
        self.fifo_out = fifo_out
        self.bytes_in = bytes_in
        self.bytes_out = bytes_out
        self.dtype = dtype
        self.tile = MyTile(0, 0)  # TODO: how to set default here?
        fifo_in.set_endpoint(self)
        fifo_out.set_endpoint(self)

        self.in_strides = None
        self.out_strides = None

        if in_sizes is None:
            self.in_sizes = [1, 1, 1, self.bytes_in]
        else:
            assert (
                len(in_sizes) > 0 and len(in_sizes) <= 4
            ), "Invalid number of in_sizes"
            assert (
                np.prod(in_sizes) == self.bytes_in
            ), "In sizes does not add up to correct input size"
            self.in_sizes = in_sizes

        if in_strides != None:
            assert (
                len(in_strides) > 0 and len(in_strides) <= 4
            ), "Invalid number of in_strides"
            self.in_strides = in_strides

        if out_sizes is None:
            self.out_sizes = [1, 1, 1, self.bytes_out]
        else:
            assert (
                len(out_sizes) > 0 and len(out_sizes) <= 4
            ), "Invalid number of out_sizes"
            assert (
                np.prod(out_sizes) == self.bytes_out
            ), "Out sizes does not add up to correct output size"
            self.out_sizes = out_sizes

        if out_strides != None:
            assert (
                len(out_strides) > 0 and len(out_strides) <= 4
            ), "Invalid number of out_strides"
            self.out_strides = out_strides

    def get_tile(self) -> MyTile:
        assert self.tile != None
        return self.tile

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return [self.fifo_in, self.fifo_out]

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        tensor_in_ty = MyTensorType.get_memref_type(
            np.ndarray[self.dtype, [self.bytes_in]]
        )
        tensor_out_ty = MyTensorType.get_memref_type(
            np.ndarray[self.dtype, [self.bytes_out]]
        )

        @runtime_sequence(tensor_in_ty, tensor_out_ty)
        def sequence(inTensor, outTensor):
            npu_dma_memcpy_nd(
                metadata=self.fifo_out.name,
                bd_id=0,
                mem=outTensor,
                sizes=self.out_sizes,
                strides=self.out_strides,
            )
            npu_dma_memcpy_nd(
                metadata=self.fifo_in.name,
                bd_id=1,
                mem=inTensor,
                sizes=self.in_sizes,
                strides=self.in_strides,
            )
            npu_sync(column=0, row=0, direction=0, channel=0)
