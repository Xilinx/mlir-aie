# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""ObjectFIFO dataflow primitives for IRON designs."""

from .objectfifo import ObjectFifo, ObjectFifoHandle, ObjectFifoLink, ObjectFifoEndpoint
from .fifo_handle_registry import (
    register_fifo_handle,
    unregister_fifo_handle,
    get_registered_handle_classes,
    dispatch_fn_arg,
)

# the original Worker.__init__ bookkeeping bit-for-bit. This is the
# backward-compat anchor: every existing IRON design that passes
# ObjectFifoHandle through fn_args still works without modification.
#
# AccumFifoHandle, SparseFifoHandle) are registered by their respective
# modules at import time, so they appear *after* ObjectFifoHandle in the
# registry and win the reverse-order isinstance() walk in dispatch_fn_arg.
def _object_fifo_handle_handler(arg, worker):
    arg.endpoint = worker
    worker._fifos.append(arg)

register_fifo_handle(ObjectFifoHandle, _object_fifo_handle_handler)
