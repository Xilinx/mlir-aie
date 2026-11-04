# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Identify the loaded IRON bindings in generated IR."""

from importlib import import_module

from .. import ir


def provenance_attr():
    """Return build metadata from this package, never from another installed wheel."""
    values = {}
    try:
        build = import_module("aie._build_version")
    except ModuleNotFoundError:
        build = None
    if build is not None:
        values.update(version=build.version, revision=build.revision, dirty=build.dirty)
    try:
        wheel = import_module("aie._version")
    except ModuleNotFoundError:
        wheel = None
    if wheel is not None:
        values["version"] = wheel.__version__
        if build is None:
            values["revision"] = wheel.__commit__
    return ir.DictAttr.get(
        {
            key: (
                ir.BoolAttr.get(value)
                if isinstance(value, bool)
                else ir.StringAttr.get(value)
            )
            for key, value in values.items()
            if value is not None
        }
    )
