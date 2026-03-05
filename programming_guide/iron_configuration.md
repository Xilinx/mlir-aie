<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# IRON Python Configurations

There are several options that exist to configure the IRON Python programming environment.

## Default IRON Tensor Class

This is a variable that controls the types of [```aie.utils.Tensor```](../python/utils/tensor.py)s that are produced by the utility functions ```tensor```, ```ones```, etc. Right now there are two tensor implementations: [```CPUOnlyTensor```](../python/utils/tensor.py) and [```XRTTensor```](../python/utils/xrtruntime/tensor.py).

By default, if ```pyxrt``` is available, the ```DEFAULT_TENSOR_CLASS``` is set to ```XRTTensor```. However, you can also manually set this value through the ```set_tensor_class()```, e.g.:
```python
>>> import numpy as np
>>> print(aie.utils.tensor.DEFAULT_TENSOR_CLASS.__name__)
XRTTensor
>>> type(iron.tensor((2, 2), np.int32))
<class 'aie.utils.xrtruntime.tensor.XRTTensor'>
>>> aie.utils.set_tensor_class(aie.utils.tensor.CPUOnlyTensor)
>>> print(aie.utils.tensor.DEFAULT_TENSOR_CLASS.__name__)
CPUOnlyTensor
>>> type(aie.utils.tensor((2, 2), np.int32))
<class 'aie.utils.tensor.CPUOnlyTensor'>
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

## IRON Cache Location

The IRON jit feature caches compiled objects in a directory defined by ```NPU_CACHE_DIR```. By default this value is the user's home directory.

## IRON XRT Runtime Cache Size

The `CachedXRTRuntime` caches XRT contexts to improve performance. The size of this cache can be configured using the `XRT_CONTEXT_CACHE_SIZE` environment variable. This is particularly useful in CI environments where multiple tests run in parallel and might exhaust the available NPU contexts.

```bash
export XRT_CONTEXT_CACHE_SIZE=1
```

## Diagnostic Output and Log Level

The `aie` library uses Python's standard `logging` module for all diagnostic output. Set
`AIE_LOG_LEVEL` to control verbosity. Valid values: `TRACE`, `DEBUG`, `INFO`, `WARNING`
(default), `ERROR`, `CRITICAL`. `TRACE` is the most verbose level, below `DEBUG`.

```bash
AIE_LOG_LEVEL=DEBUG python my_script.py    # show debug messages
AIE_LOG_LEVEL=TRACE python my_script.py    # show all messages including trace-level
AIE_LOG_LEVEL=ERROR python my_script.py    # errors only
```

For per-module control or routing to a file, use the `logging` API directly:

```python
import logging

logging.getLogger("aie").setLevel(logging.ERROR)

# Route aie logs to a file instead of the console
handler = logging.FileHandler("aie.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logging.getLogger("aie").addHandler(handler)
logging.getLogger("aie").propagate = False  # don't also send to root logger
```
