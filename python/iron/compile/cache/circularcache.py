# cache.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
from pathlib import Path
import shutil
import fcntl
import contextlib
import time

# The `iron.compiledesign` decorator below caches compiled kenrels inside the `IRON_CACHE_HOME` directory.
# Kernels are cached based on their hash value of the MLIR module string. If during compilation,
# we hit in the cache, the `iron.jit` will load the xclbin and instruction binary files from the cache.
IRON_CACHE_HOME = os.environ.get("IRON_CACHE_HOME", Path.home() / ".iron" / "cache")


class CircularCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = [None] * max_size
        self.keys = [None] * max_size
        self.index = 0

    def __contains__(self, key):
        return key in self.keys

    def __getitem__(self, key):
        idx = self.keys.index(key)
        return self.cache[idx]

    def __setitem__(self, key, value):
        self.cache[self.index] = value
        self.keys[self.index] = key
        self.index = (self.index + 1) % self.max_size

    def __len__(self):
        return sum(1 for k in self.keys if k is not None)

    def clear(self):
        self.cache = [None] * self.max_size
        self.keys = [None] * self.max_size
        self.index = 0
