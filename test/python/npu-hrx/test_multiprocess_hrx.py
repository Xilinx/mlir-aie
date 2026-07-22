# test_multiprocess_hrx.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: hrx_python_bindings

"""Multi-process HRX dispatch (review comment r3623840141).

Covers the "several python processes at once / multiple users" question: each
process builds its own :class:`~aie.utils.hostruntime.hrxruntime.context.HRXContext`
(its own device/stream) and its own buffers, so the amdxdna driver isolates
them. This spawns several independent processes that each compile + dispatch an
HRX design concurrently (sharing one ``NPU_CACHE_HOME``, which also exercises
the compile-cache file locking) and asserts they all succeed -- i.e. concurrent
processes neither corrupt each other's buffers nor deadlock on the shared,
finite amdxdna hardware-context pool.

Adapted from ``test/python-concurrency/jit_parallel_compilation.py``.
"""

import os
import subprocess
import sys
import tempfile

import pytest

# Each subprocess: build an `out = in + 1` design, dispatch it through the HRX
# runtime (IRON_RUNTIME=hrx forces CachedHRXRuntime), and verify the result.
_WORKER_SCRIPT = """
import sys
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_

_TILE = 16
_SIZE = 1024


@iron.jit
def add_one(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE):
                elem_out[i] = elem_in[i] + 1
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, b):
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


try:
    import aie.utils as u
    assert u.has_hrx, "libhrx not discoverable in subprocess"

    base = np.arange(1, _SIZE + 1, dtype=np.int32)
    in_a = iron.tensor(base, dtype=np.int32, device="npu")
    out = iron.zeros(_SIZE, dtype=np.int32, device="npu")

    add_one(in_a, out, N=_SIZE)

    out.to("cpu")
    np.testing.assert_array_equal(out.numpy(), base + 1)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
"""


def test_multiprocess_hrx_dispatch():
    """Several processes dispatch HRX designs concurrently; all must succeed."""
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        script_path = os.path.join(temp_cache_dir, "hrx_worker.py")
        with open(script_path, "w") as f:
            f.write(_WORKER_SCRIPT)

        num_processes = 4
        processes = []
        for _ in range(num_processes):
            env = os.environ.copy()
            env["NPU_CACHE_HOME"] = temp_cache_dir
            env["IRON_RUNTIME"] = "hrx"
            processes.append(
                subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
            )

        return_codes = []
        outputs = []
        for i, process in enumerate(processes):
            stdout, stderr = process.communicate()
            return_codes.append(process.returncode)
            outputs.append((stdout, stderr))
            print(f"\n=== HRX process {i} (return code: {process.returncode}) ===")
            print(f"STDOUT:\n{stdout}")
            print(f"STDERR:\n{stderr}")
            print("=" * 50)

        successful = sum(1 for code in return_codes if code == 0)
        assert len(return_codes) == num_processes, "All processes should complete"

        if successful < num_processes:
            msg = f"Only {successful}/{num_processes} HRX processes succeeded\n"
            for i, (code, (out, err)) in enumerate(zip(return_codes, outputs)):
                status = "SUCCESS" if code == 0 else "FAILED"
                msg += f"\nProcess {i}: {status} (return code: {code})\n"
                if out:
                    msg += f"  STDOUT: {out.strip()}\n"
                if err:
                    msg += f"  STDERR: {err.strip()}\n"
            pytest.fail(msg)
