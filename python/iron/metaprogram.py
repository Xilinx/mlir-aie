# metaprogram.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import contextvars

_metaprograms = contextvars.ContextVar("metaprograms", default={})


class metaprogramming_ctx:
    def __init__(self, **kwargs):
        self.metaprograms = kwargs
        self._old_metaprograms = None

    def __enter__(self):
        self._old_metaprograms = _metaprograms.get()
        _metaprograms.set({**self._old_metaprograms, **self.metaprograms})
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _metaprograms.set(self._old_metaprograms)


def get_metaprogram(key):
    return _metaprograms.get().get(key)
