"""
TODO: 
* docs
* validation...?
"""

import numpy as np

from .... import ir
from ....dialects.aiex import runtime_sequence, npu_sync, npu_dma_memcpy_nd
from .inout import InOutSequence
from ..objectfifo import ObjectFifoHandle
from ....extras.util import np_ndarray_type_to_mlir_type, get_np_ndarray_type_shape


class SimpleFifoInOutSequence(InOutSequence):
    def __init__(
        self,
        fifo_in: ObjectFifoHandle,
        bytes_in: int,
        fifo_out: ObjectFifoHandle,
        bytes_out: int,
        in_sizes: list[int] | None = None,
        in_strides: list[int] | None = None,
        out_sizes: list[int] | None = None,
        out_strides: list[int] | None = None,
        dtype: np.generic = np.uint8,
    ):
        assert bytes_in % np.prod(get_np_ndarray_type_shape(fifo_in.obj_type)) == 0
        assert bytes_out % np.prod(get_np_ndarray_type_shape(fifo_in.obj_type)) == 0
        assert bytes_in > 0
        assert bytes_out > 0
        # TODO: make sure fifo endpoint is a shim tile

        self.fifo_in = fifo_in
        self.fifo_out = fifo_out
        self.bytes_in = bytes_in
        self.bytes_out = bytes_out
        self.dtype = dtype

        self.in_strides = None
        self.out_strides = None

        if in_sizes is None:
            self.in_sizes = [1, 1, 1, self.bytes_in]
        else:
            assert (
                len(in_sizes) > 0 and len(in_sizes) <= 4
            ), "Invalid number of in_sizes"
            assert (
                self.bytes_in % np.prod(in_sizes) == 0
            ), "in_sizes does not correctly divide input size"
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
                self.bytes_out % np.prod(out_sizes) == 0
            ), "out_sizes does not correctly divide output size"
            self.out_sizes = out_sizes

        if out_strides != None:
            assert (
                len(out_strides) > 0 and len(out_strides) <= 4
            ), "Invalid number of out_strides"
            self.out_strides = out_strides

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return [self.fifo_in, self.fifo_out]

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        tensor_in_ty = np_ndarray_type_to_mlir_type(
            np.ndarray[self.dtype, (self.bytes_in,)]
        )
        tensor_out_ty = np_ndarray_type_to_mlir_type(
            np.ndarray[self.dtype, (self.bytes_out,)]
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
