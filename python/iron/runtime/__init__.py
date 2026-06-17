# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Runtime data-movement internals.

The runtime sequence is supplied directly to :class:`aie.iron.Program` as a
callable and resolved by it; there is no user-facing ``Runtime`` object. The
modules here (``_sequence``, ``_context``, ``taskgroup``, ``dmatask``, ``data``)
are internal machinery reached during ``Program.resolve_program``.
"""
