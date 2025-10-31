# metaprogram.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import contextvars
from typing import Any

_compile_args = contextvars.ContextVar("compile_args", default={})


class compile_ctx:
    """A context manager for compile arguments."""

    def __init__(self, **kwargs):
        """Initializes the compile_ctx object.

        Args:
            **kwargs: The compile arguments.
        """
        self.metaargs = kwargs
        self._old_metaargs = None

    def __enter__(self):
        """Enters the context."""
        self._old_metaargs = _compile_args.get()
        _compile_args.set({**self._old_metaargs, **self.metaargs})
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context."""
        _compile_args.set(self._old_metaargs)


def get_compile_arg(key: str) -> Any:
    """Gets a compile argument.

    Args:
        key (str): The key of the compile argument.

    Returns:
        Any: The value of the compile argument.
    """
    return _compile_args.get().get(key)
