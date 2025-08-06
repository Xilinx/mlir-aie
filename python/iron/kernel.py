# kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
import cxxfilt
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from .. import ir  # type: ignore
from ..extras.dialects.ext.func import FuncOp  # type: ignore
from ..helpers.dialects.ext.func import call
from ..dialects.aie import external_func
from .resolvable import Resolvable

def find_manged_symbol(file, demangled_name):
    """
    Find the mangled symbol that corresponds to the demangled_name.

    Args:
        file (str): Path to the file to analyze
        demangled_name (str): The demangled name of the symbol to find

    Returns:
        str: The mangled name of the symbol if found, otherwise None
    """
    with open(file, 'rb') as file:
        elf_file = ELFFile(file)

        for section in elf_file.iter_sections():
            if isinstance(section, SymbolTableSection):
                for symbol in section.iter_symbols():
                    # Filter out function symbols
                    if symbol and symbol['st_info']['type'] == 'STT_FUNC':
                        if symbol.name == demangled_name:
                            # Name matches the demangled name, thus it has C linkage
                            return symbol.name
                        if cxxfilt.demangle(symbol.name) == demangled_name:
                            # Demangled symbol name matches the demangled name, thus it has C++ linkage
                            return symbol.name
    return None

class Kernel(Resolvable):
    def __init__(
        self,
        name: str,
        bin_name: str,
        arg_types: list[type[np.ndarray] | np.dtype] = [],
    ) -> None:
        """A Kernel is an externally defined function that eventually resolves to a FuncOp. If it is called,
        a CallOp will be generated.

        Args:
            name (str): The name of the function
            bin_name (str): The name of the binary (used for linking to a compute core)
            arg_types (list[type[np.ndarray]  |  np.dtype], optional): The type signature of the function. Defaults to [].
        """

        symbol_name = find_manged_symbol(f"build/{bin_name}", name)
        if not symbol_name:
            raise ValueError(f"Could not find symbol for {name} in {bin_name}")

        self._name = symbol_name
        self._bin_name = bin_name
        self._arg_types = arg_types
        self._op: FuncOp | None = None

    @property
    def bin_name(self) -> str:
        return self._bin_name

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._op:
            self._op = external_func(self._name, inputs=self._arg_types)

    def __call__(self, *args, **kwargs):
        if not self._op:
            raise ValueError("Need to resolve Kernel before it can be called")
        call(self._op, args, **kwargs)
