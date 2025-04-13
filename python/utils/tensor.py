# tensor.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
import pyxrt as xrt
import ctypes


class Tensor:
    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device='{self.device}')\n{self.data}"

    def __init__(self, shape_or_data, dtype=np.uint32, device="npu"):
        if device != "npu" and device != "cpu":
            raise ValueError("Unsupported device: {}".format(device))

        self.device = device

        if isinstance(shape_or_data, tuple):
            self.shape = shape_or_data
            self.dtype = dtype
            self.data = np.zeros(self.shape, dtype=self.dtype)
        else:
            np_data = np.array(shape_or_data, dtype=dtype)
            self.shape = np_data.shape
            self.dtype = np_data.dtype
            self.data = np_data.copy()

        self.len_bytes = np.prod(self.shape) * np.dtype(self.dtype).itemsize
        device_index = 0
        self.xrt_device = xrt.device(device_index)

        # Ideally, we use xrt::ext::bo host-only BO but there are no bindings for that currenty.
        # Eventually, xrt:ext::bo uses the 0 magic number that shall be fixed in the future.
        # https://github.com/Xilinx/XRT/blob/9b114f18c4fcf4e3558291aa2d78f6d97c406365/src/runtime_src/core/common/api/xrt_bo.cpp#L1626
        group_id = 0
        self.bo = xrt.bo(
            self.xrt_device,
            self.len_bytes,
            xrt.bo.host_only,
            group_id,
        )
        ptr = self.bo.map()
        self.data = np.frombuffer(ptr, dtype=self.dtype).reshape(self.shape)

        if not isinstance(shape_or_data, tuple):
            np.copyto(self.data, np_data)
            self.__sync_to_device()

    def __array__(self, dtype=None):
        self.__sync_from_device()
        if dtype:
            return self.data.astype(dtype)
        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def to(self, target_device: str):
        if target_device == "npu":
            self.__sync_to_device()
            return self
        elif target_device == "cpu":
            self.__sync_from_device()
            return self
        else:
            raise ValueError(f"Unknown device '{target_device}'")

    def __sync_to_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def __sync_from_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def buffer_object(self):
        return self.bo

    def numpy(self):
        self.__sync_from_device()
        return self.data

    def numel(self):
        return int(np.prod(self.shape))

    @classmethod
    def ones(cls, shape, dtype=np.uint32, device="cpu", **kwargs):
        t = cls(shape, dtype, device, **kwargs)
        t.data.fill(1)
        if t.device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def zeros(cls, shape, dtype=np.uint32, device="cpu", **kwargs):
        t = cls(shape, dtype, device, **kwargs)
        t.data.fill(0)
        if t.device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def randomints(cls, shape, low, high, dtype=np.uint32, device="cpu", **kwargs):
        t = cls(shape, dtype, device, **kwargs)
        t.data[:] = np.random.randint(low, high, size=shape, dtype=dtype)
        if t.device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def random(cls, shape, low=0.0, high=1.0, dtype=np.float32, device="cpu", **kwargs):
        t = cls(shape, dtype, device, **kwargs)
        t.data[:] = np.random.uniform(low, high, size=shape).astype(dtype)
        if t.device == "npu":
            t.__sync_to_device()
        return t

    @staticmethod
    def _ctype_from_dtype(dtype):
        if dtype == np.uint32:
            return ctypes.c_uint32
        elif dtype == np.int32:
            return ctypes.c_int32
        elif dtype == np.float32:
            return ctypes.c_float
        else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")

    @classmethod
    def arange(cls, start, stop=None, step=1, dtype=np.uint32, device="cpu", **kwargs):
        if stop is None:
            start, stop = 0, start

        data = np.arange(start, stop, step, dtype=dtype)
        shape = data.shape

        t = cls(shape, dtype, device, **kwargs)
        t.data[...] = data
        if t.device == "npu":
            t.__sync_to_device()
        return t

    @classmethod
    def zerolike(cls, other, dtype=None, device=None, **kwargs):
        dtype = dtype or other.dtype
        device = device or other.device
        t = cls(other.shape, dtype=dtype, device=device, **kwargs)
        t.data.fill(0)
        if t.device == "npu":
            t.__sync_to_device()
        return t

    def __del__(self):
        del self.bo
        self.bo = None


def tensor(data):
    return Tensor(data)


ones = Tensor.ones
zeros = Tensor.zeros
random = Tensor.random
randomints = Tensor.randomints
arange = Tensor.arange
zerolike = Tensor.zerolike
