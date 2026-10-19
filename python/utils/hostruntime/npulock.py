# npulock.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""An inter-process lock on the NPU, held only while this process has a hardware context.

Processes that share an NPU and set ``NPU_LOCK`` to the same file take turns on
the device, while compiling and other host work proceeds unlocked. Each runtime
takes `npu_lock` before it creates a hardware context and releases it in
``cleanup()``; process exit releases it too.
"""

import logging
import os
import threading
import time
import weakref

if os.name != "nt":
    import fcntl

logger = logging.getLogger(__name__)


class NPULock:
    """An exclusive ``flock`` on `path`, held while any holder has a hardware context.

    Args:
        path: The lock file, created if missing. ``None`` disables the lock.

    Raises:
        NotImplementedError: `path` is set on Windows, which has no ``flock``.
    """

    def __init__(self, path: str | None):
        if path is not None and os.name == "nt":
            raise NotImplementedError("NPU_LOCK needs flock, which Windows lacks")
        self.path = path
        self._fd: int | None = None
        self._holders: weakref.WeakSet = weakref.WeakSet()
        self._mutex = threading.Lock()

    @property
    def held(self) -> bool:
        """Whether this process holds the lock."""
        return self._fd is not None

    def acquire(self, holder: object) -> None:
        """Count `holder` as having a hardware context, first taking the lock if no holder has.

        Blocks, with no timeout, while another process holds the lock.

        Args:
            holder: The runtime or context about to create a hardware context.
        """
        if self.path is None:
            return
        with self._mutex:
            if self._fd is None:
                fd = os.open(self.path, os.O_RDONLY | os.O_CREAT, 0o666)
                start = time.monotonic()
                try:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        logger.info("Waiting for the NPU lock %s", self.path)
                        fcntl.flock(fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(fd)
                    raise
                logger.info(
                    "Took the NPU lock %s after %.1f s",
                    self.path,
                    time.monotonic() - start,
                )
                self._fd = fd
            self._holders.add(holder)

    def release(self, holder: object) -> None:
        """Stop counting `holder`, and release the lock once no holder is left.

        Args:
            holder: A runtime or context that has freed its hardware contexts.
        """
        with self._mutex:
            self._holders.discard(holder)
            if self._fd is None or self._holders:
                return
            # LOCK_UN, not just close: a forked child shares this open file.
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
            logger.info("Released the NPU lock %s", self.path)


npu_lock = NPULock(os.environ.get("NPU_LOCK"))
"""This process's lock, on the file ``NPU_LOCK`` names; disabled when it is unset."""
