# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class KernelHandle:
    pass


class HostRuntime(ABC):
    """An abstract class for a generic host runtime"""

    @abstractmethod
    def load(self, *args, **kwargs) -> KernelHandle:
        pass

    @abstractmethod
    def run(self, kernel_handle: KernelHandle, *args, **kwargs):
        pass

    def load_and_run(self, load_args: list, run_args: list):
        handle = self.load(load_args)
        self.run(handle, run_args)

    @abstractmethod
    def device_str(self) -> str:
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up all resources"""
        pass

    @abstractmethod
    def reset(self):
        """Reset the runtime"""
        pass
