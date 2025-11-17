<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# IRON Python Binding Configurations

There are several options that exist to configure the IRON Python programming environment.

## Default IRON Tensor Class

This is a variable that controls the types of [```iron.Tensor```s](../python/iron/hostruntime/tensor.py) that are produced by the utility functions ```tensor```, ```ones```, etc. Right now there are two tensor implementations: [```CPUOnlyTensor```](../python/iron/hostruntime/tensor.py) and [```XRTTensor```](../python/iron/hostruntime/xrtruntime/tensor.py).

By default, if ```pyxrt``` is available, the ```DEFAULT_IRON_TENSOR_CLASS``` is set to ```XRTTensor```. However, you can also manually set this value through the ```set_iron_tensor_class()```, e.g.:
```python
>>> import aie.iron as iron
>>> import numpy as np
>>> print(iron.hostruntime.tensor.DEFAULT_IRON_TENSOR_CLASS.__name__)
XRTTensor
>>> type(iron.tensor((2, 2), np.int32))
<class 'aie.iron.hostruntime.xrtruntime.tensor.XRTTensor'>
>>> iron.set_iron_tensor_class(iron.hostruntime.tensor.CPUOnlyTensor)
>>> print(iron.hostruntime.tensor.DEFAULT_IRON_TENSOR_CLASS.__name__)
CPUOnlyTensor
>>> type(iron.tensor((2, 2), np.int32))
<class 'aie.iron.hostruntime.tensor.CPUOnlyTensor'>
```

## Default IRON Device

If the IRON device is not set, many designs will try it fetch it on demand using the utility function [```detect_npu_device()```](../python/iron/hostruntime/config.py). However, this can be overriden by calling the [```set_current_device()```](../python/iron/hostruntime/config.py) function, which takes as an argument the new device and returns the previous device:
```python
>>> import aie.iron as iron
>>> iron.set_current_device(iron.device.NPU1())
<abc.NPU2 object at 0x722a659826c0>
>>> iron.get_current_device()
<abc.NPU1 object at 0x722a65903a10>
```

