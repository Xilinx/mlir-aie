from .globalbuffer import GlobalBuffer
from .kernel import ExternalFunction, Kernel
from .localbuffer import LocalBuffer
from .program import Program
from .worker import Worker, WorkerRuntimeBarrier
from .runtime import Runtime
from .dataflow import ObjectFifo
from .dtype import str_to_dtype, dtype_to_str

try:
    # The .jit module requires pyxrt which may not exist.
    # The try-except block tests if the module exists and
    # imports the .jit module.
    import pyxrt
    from .jit import jit
    from .config import set_current_device, get_current_device, detect_npu_device

    from .tensor import (
        tensor,
        ones,
        zeros,
        randint,
        rand,
        arange,
        zeros_like,
    )
except ImportError:
    pass  # silently ignore if pyxrt or .jit can't be imported
