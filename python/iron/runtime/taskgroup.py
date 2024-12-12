# taskgroup.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.


class RuntimeTaskGroup:
    """A RuntimeTaskGroup is a structured tag to indicated groupings of RuntimeTasks."""

    def __init__(self, id: int):
        """Construct a RuntimeTaskGroup

        Args:
            id (int): The id of the task group. The id should be unique to tasks groups within a Runtime.
        """
        self._group_id = id

    @property
    def group_id(self) -> int:
        """The id of the task group."""
        return self._group_id

    def __str__(self):
        return f"TaskGroup({self.group_id})"
