# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Hardware-free ownership and capacity tests for dynamic runtime instructions."""

import importlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fake_xrt(monkeypatch):
    fake = SimpleNamespace(
        bo=SimpleNamespace(cacheable=1, host_only=2),
        ert_cmd_state=SimpleNamespace(ERT_CMD_STATE_COMPLETED=0),
        xrt_info_device=SimpleNamespace(name=0),
    )
    prefix = "aie.utils.hostruntime.xrtruntime"
    before = {name for name in sys.modules if name.startswith(prefix)}
    monkeypatch.setitem(sys.modules, "pyxrt", fake)
    hrt = importlib.import_module(prefix + ".hostruntime")
    monkeypatch.setattr(hrt, "pyxrt", fake)
    monkeypatch.setattr(
        hrt, "acquire_device", lambda: SimpleNamespace(get_info=lambda key: "Strix")
    )
    monkeypatch.setattr(
        hrt.XRTHostRuntime, "check_device_consistency", lambda self: None
    )
    allocations = []

    def hook(phase, data):
        pass

    class Transport:
        def __init__(self, device, nbytes, flags, group):
            self.nbytes = nbytes
            self.host_bytes = np.zeros(nbytes, dtype=np.uint8)
            allocations.append(self)

        def root_prefix_to_device(self, nbytes):
            hook("sync", self.host_bytes[:nbytes])
            return self

    monkeypatch.setattr(hrt, "XrtTransport", Transport)

    def make_kernel(callback):
        nonlocal hook
        hook = callback

        class Kernel:
            def group_id(self, index):
                return 1

            def __call__(self, opcode, bo, nbytes, *buffers):
                data = bo.host_bytes[:nbytes]
                callback("submit", data)

                def wait():
                    callback("wait", data)
                    return 0

                return SimpleNamespace(wait=wait)

        return hrt.XRTKernelHandle(
            Kernel(), SimpleNamespace(get_kernels=lambda: []), None, None
        )

    try:
        yield hrt, allocations, make_kernel
    finally:
        # Do not leave modules bound to the fake pyxrt for other test files.
        for name in sorted(sys.modules, reverse=True):
            if name.startswith(prefix) and name not in before:
                module = sys.modules.pop(name)
                parent_name, _, attr = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, attr, None) is module:
                    delattr(parent, attr)


@pytest.mark.parametrize("blocked_phase", ["sync", "submit", "wait"])
def test_xrt_dynamic_buffer_owned_through_wait(fake_xrt, blocked_phase):
    hrt, allocations, make_kernel = fake_xrt
    rt = hrt.XRTHostRuntime()
    blocked = threading.Event()
    release = threading.Event()
    contender = threading.Event()
    observations = []
    failures = []
    real_lock = rt._dispatch_lock

    class ObservedLock:
        def acquire(self):
            if threading.current_thread().name == "second":
                contender.set()
            return real_lock.acquire()

        def release(self):
            real_lock.release()

    rt._dispatch_lock = ObservedLock()

    def callback(phase, data):
        name = threading.current_thread().name
        if name == "first" and phase == blocked_phase:
            blocked.set()
            assert release.wait(5)
        observations.append((name, phase, data.view(np.uint32).tolist()))

    handle = make_kernel(callback)

    def run(value):
        try:
            result = rt.run(
                handle, [], dispatch_insts=np.array([value], dtype=np.uint32)
            )
            assert result.is_success()
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=run, args=(11,), name="first")
    second = threading.Thread(target=run, args=(22,), name="second")
    first.start()
    try:
        assert blocked.wait(5)
        second.start()
        assert contender.wait(5), "second dispatch must acquire the ownership lock"
        assert not any(name == "second" for name, _, _ in observations)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert observations == [
        (name, phase, [value])
        for name, value in [("first", 11), ("second", 22)]
        for phase in ["sync", "submit", "wait"]
    ]
    assert len(allocations) == 1


@pytest.mark.parametrize("failure_phase", ["sync", "submit", "wait"])
def test_xrt_dynamic_lock_released_on_exception(fake_xrt, failure_phase):
    hrt, _, make_kernel = fake_xrt
    rt = hrt.XRTHostRuntime()

    def fail(phase, data):
        if phase == failure_phase:
            raise RuntimeError("simulated XRT failure")

    handle = make_kernel(fail)
    words = np.array([1], dtype=np.uint32)
    with pytest.raises(RuntimeError, match="simulated XRT failure"):
        rt.run(handle, [], dispatch_insts=words)
    assert rt._dispatch_lock.acquire(blocking=False)
    rt._dispatch_lock.release()
    handle = make_kernel(lambda phase, data: None)
    assert rt.run(handle, [], dispatch_insts=words).is_success()


def test_xrt_dynamic_reuse_growth_and_static_path(fake_xrt):
    hrt, allocations, make_kernel = fake_xrt
    rt = hrt.XRTHostRuntime()
    sizes = []
    handle = make_kernel(lambda phase, data: sizes.append((phase, len(data))))
    for count in (4, 2, 8, 1):
        assert rt.run(
            handle, [], dispatch_insts=np.arange(count, dtype=np.uint32)
        ).is_success()
    assert [storage.nbytes for storage in allocations] == [16, 32]
    assert [nbytes for phase, nbytes in sizes if phase == "submit"] == [16, 8, 32, 4]
    handle.kernel.group_id = lambda index: 2
    assert rt.run(
        handle, [], dispatch_insts=np.array([1], dtype=np.uint32)
    ).is_success()
    assert len(allocations) == 3

    class ForbiddenLock:
        def acquire(self):
            raise AssertionError(
                "static instructions must not acquire the dynamic lock"
            )

    rt._dispatch_lock = ForbiddenLock()
    handle.insts = np.array([7], dtype=np.uint32)
    handle.insts_bo = SimpleNamespace(host_bytes=handle.insts.view(np.uint8))
    assert rt.run(handle, []).is_success()
    assert len(allocations) == 3


@pytest.fixture
def fake_hrx(monkeypatch):
    from aie.utils.hostruntime.hrxruntime import hostruntime as hrt

    class Context:
        def __init__(self):
            self.limit = 2
            self.refs = {}
            self.events = []
            self.next_exe = 0
            self.fail = None

        def create_executable(self, image, insts, name):
            assert len(self.refs) < self.limit, "driver hardware contexts exhausted"
            if self.fail == "create":
                raise hrt.HRXError("create")
            self.next_exe += 1
            exe = self.next_exe
            self.refs[exe] = 1
            self.events.append(("create", exe, image, insts))
            return exe

        def lookup_export(self, exe, name):
            if self.fail == "lookup":
                raise hrt.HRXError("lookup")
            return 0

        def retain_executable(self, exe):
            self.refs[exe] += 1

        def release_executable(self, exe):
            self.events.append(("release", exe))
            self.refs[exe] -= 1
            if not self.refs[exe]:
                del self.refs[exe]

        def dispatch(self, exe, ordinal, bindings):
            assert exe in self.refs
            self.events.append(("dispatch", exe))
            if self.fail == "dispatch":
                raise hrt.HRXError("dispatch")

        def synchronize(self):
            self.events.append(("sync",))
            if self.fail == "sync":
                raise hrt.HRXError("sync")

    ctx = Context()
    monkeypatch.setattr(hrt.HRXContext, "get", classmethod(lambda cls: ctx))
    monkeypatch.setattr(hrt, "_detect_hrx_device_gen", lambda: "npu1")
    monkeypatch.setattr(
        hrt.HRXHostRuntime, "check_device_consistency", lambda self: None
    )
    monkeypatch.setattr(
        hrt.atexit, "register", lambda callback, *args, **kwargs: None
    )
    monkeypatch.setenv("HRX_EXE_CACHE_SIZE", "2")
    rt = hrt.CachedHRXRuntime()
    try:
        yield hrt, rt, ctx
    finally:
        rt.cleanup()


def _kernel_files(directory, name, dynamic=False):
    xclbin = directory / (name + ".xclbin")
    xclbin.write_bytes(b"xclbin:" + name.encode())
    insts = directory / (name + ".bin")
    if not dynamic:
        insts.write_bytes(b"\x01\x00\x00\x00")
    return SimpleNamespace(
        xclbin_path=xclbin,
        insts_path=None if dynamic else insts,
        kernel_name="MLIR_AIE",
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("limit", [6, 16])
def test_hrx_reclaims_before_creating_at_capacity(fake_hrx, tmp_path, dynamic, limit):
    _, rt, ctx = fake_hrx
    rt._cache_size = ctx.limit = limit
    kernels = [_kernel_files(tmp_path, str(i)) for i in range(limit)]
    for kernel in kernels:
        handle = rt.load(kernel)
        del handle
    assert len(ctx.refs) == limit
    ctx.events.clear()
    kernel = _kernel_files(tmp_path, "next", dynamic=dynamic)
    handle = rt.load(kernel)
    if dynamic:
        assert rt.run(
            handle, [], dispatch_insts=np.array([23], dtype=np.uint32)
        ).is_success()
    assert ctx.events[0] == ("release", 1)
    assert ctx.events[1][0:2] == ("create", limit + 1)
    assert len(ctx.refs) == (limit - 1 if dynamic else limit)
    del handle


def test_hrx_dynamic_image_cached_across_loads(fake_hrx, tmp_path, monkeypatch):
    _, rt, ctx = fake_hrx
    kernel = _kernel_files(tmp_path, "dynamic", dynamic=True)
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    for word in (11, 22):
        handle = rt.load(kernel)
        assert rt.run(
            handle, [], dispatch_insts=np.array([word], dtype=np.uint32)
        ).is_success()
    assert reads == [kernel.xclbin_path]
    creates = [event for event in ctx.events if event[0] == "create"]
    assert [event[3] for event in creates] == [
        np.array([word], dtype=np.uint32).tobytes() for word in (11, 22)
    ]
    assert not ctx.refs

    previous = handle
    stat = kernel.xclbin_path.stat()
    monkeypatch.setattr(
        Path, "stat", lambda path: SimpleNamespace(st_mtime=stat.st_mtime + 1)
    )
    monkeypatch.setattr(
        rt, "_resolve_kernel", lambda kernel: (kernel.xclbin_path, None, "MLIR_AIE")
    )
    assert rt.load(kernel) is not previous
    rt.cleanup()
    assert not rt._dispatch_cache


def test_hrx_static_cache_hits_and_lru_ownership(fake_hrx, tmp_path):
    _, rt, ctx = fake_hrx
    kernels = [_kernel_files(tmp_path, str(i)) for i in range(3)]
    for kernel in kernels[:2]:
        handle = rt.load(kernel)
        del handle
    handle = rt.load(kernels[0])
    assert handle.executable == 1
    assert ctx.next_exe == 2
    replacement = rt.load(kernels[2])
    assert set(ctx.refs) == {1, 3}, "cache hit must move executable 1 to MRU"
    del replacement
    rt.cleanup()
    assert ctx.refs == {1: 1}, "cache cleanup must preserve a live handle's reference"
    assert rt.run(handle, []).is_success()
    del handle
    assert not ctx.refs


def test_hrx_dynamic_image_cache_is_bounded(fake_hrx, tmp_path):
    _, rt, ctx = fake_hrx
    kernels = [_kernel_files(tmp_path, str(i), dynamic=True) for i in range(3)]
    first = rt.load(kernels[0])
    for kernel in kernels[1:]:
        rt.load(kernel)
    assert len(rt._dispatch_cache) == 2
    assert rt.load(kernels[0]) is not first
    assert not ctx.refs


@pytest.mark.parametrize("failure", ["create", "lookup", "dispatch", "sync"])
@pytest.mark.parametrize("fail_on_error", [False, True])
def test_hrx_dynamic_executable_failure_cleanup(
    fake_hrx, tmp_path, failure, fail_on_error
):
    hrt, rt, ctx = fake_hrx
    handle = rt.load(_kernel_files(tmp_path, "dynamic", dynamic=True))
    ctx.fail = failure
    kwargs = dict(
        dispatch_insts=np.array([1], dtype=np.uint32), fail_on_error=fail_on_error
    )
    if failure in ("create", "lookup") or fail_on_error:
        with pytest.raises(hrt.HostRuntimeError, match=failure):
            rt.run(handle, [], **kwargs)
    else:
        assert not rt.run(handle, [], **kwargs).is_success()
    assert not ctx.refs


def test_hrx_zero_cache_and_live_handle_ownership(fake_hrx, tmp_path):
    _, rt, ctx = fake_hrx
    rt._cache_size = 0
    handle = rt.load(_kernel_files(tmp_path, "static"))
    assert not rt._exe_cache
    assert ctx.refs == {1: 2}
    rt.cleanup()
    assert ctx.refs == {1: 1}
    assert rt.run(handle, []).is_success()
    del handle
    assert not ctx.refs
