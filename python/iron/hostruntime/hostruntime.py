# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from ..device import Device


class KernelHandle:
    pass


class HostRuntime(ABC):
    """An abstract class for a generic host runtime"""

    @abstractmethod
    def load(self, *args, **kwargs) -> KernelHandle:
        pass

    @abstractmethod
    def run(self, kernel_handle: KernelHandle, *args):
        pass

    def load_and_run(self, load_args: list, run_args: list):
        handle = self.load(load_args)
        self.run(handle, run_args)

    @abstractmethod
    def device(self) -> Device:
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up all resources"""
        pass

    @abstractmethod
    def reset(self):
        """Reset the runtime"""
        pass


class IronRuntimeError(Exception):
    """
    Error raised when a NPU kernel encounters an error during runtime operations.
    """

    pass


# Set default tensor class
try:
    from .xrtruntime.hostruntime import XRTHostRuntime

    # For now we assume runtimes are singletons, and go ahead and instantiate here.
    DEFAULT_IRON_RUNTIME = XRTHostRuntime()
except ImportError:
    DEFAULT_IRON_RUNTIME = None
