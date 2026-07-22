# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Runtime: host-side data movement and worker execution orchestration."""

from .runtime import Runtime, sync_parameters
from .taskgroup import TaskGroup
from .data import RuntimeData
from .dmataskhandle import Task
