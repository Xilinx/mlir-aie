# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Only declared whole-call trace regions may become benchmark cycle metrics."""

from types import SimpleNamespace

import numpy as np
import pytest
from aie.iron import ExternalFunction, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU2Col1
from aie.iron.kernels import KernelContract, Trace
from aie.utils.compile.jit import InOut
from aie.utils.hostruntime import set_current_device


def _kernel(name, contract, arg_types=()):
    kernel = ExternalFunction(
        name, source_string=f'extern "C" void {name}() {{}}', arg_types=list(arg_types)
    )
    kernel.contract = contract
    return kernel


@pytest.fixture
def fn():
    return _kernel("profiled", KernelContract(roles=(), trace=Trace.whole_call()))


def _run(fn, tmp_path, calls=1):
    return kd.cycles_per_call(
        None, [], 1, int, fn=fn, trace_size=1024, workdir=tmp_path, calls=calls
    )


@pytest.mark.parametrize(
    "trace", [Trace.none("no markers"), Trace.partial("markers wrap a band")]
)
def test_an_untimed_contract_returns_its_reason_without_running(tmp_path, trace):
    fn = _kernel("untimed", KernelContract(roles=(), trace=trace))
    assert _run(fn, tmp_path) == kd.CallCycles(untimed=trace.reason)
    assert kd.traced_intervals(fn, calls=4) == 0


def test_an_undeclared_trace_is_refused(tmp_path):
    fn = _kernel("undeclared", KernelContract(roles=()))
    with pytest.raises(ValueError, match="declares no trace"):
        _run(fn, tmp_path)


def _with_initializer(trace):
    zero = _kernel("zero_acc", KernelContract(roles=(), trace=trace))
    return _kernel(
        "acc",
        KernelContract(
            roles=(InOut,),
            trace=Trace.whole_call(),
            initializers=((0, lambda _fn: zero),),
        ),
        [np.ndarray[(16,), np.dtype[np.int32]]],
    )


def test_a_traced_initializer_is_one_more_interval_per_call():
    assert kd.traced_intervals(_with_initializer(Trace.whole_call()), calls=4) == 8
    assert (
        kd.traced_intervals(_with_initializer(Trace.none("no markers")), calls=4) == 4
    )


def test_the_library_mm_is_zeroed_then_timed_each_call():
    # zero's markers bracket its call, so every mm call emits two intervals;
    # set_rounding, the eltwise setup, emits none.
    set_current_device(NPU2Col1())
    try:
        assert kd.traced_intervals(kernels.mm(), calls=4) == 8
        assert kd.traced_intervals(kernels.add(), calls=4) == 4
    finally:
        set_current_device(None)


def test_a_partial_initializer_cannot_be_split_off(tmp_path):
    fn = _with_initializer(Trace.partial("markers wrap each row"))
    with pytest.raises(ValueError, match="markers wrap each row"):
        _run(fn, tmp_path)


# The stream mm's harness emits: zero, then the kernel, per call.
def test_split_labels_each_call_by_position():
    setup, (zero, mm), truncated = kd.split_intervals(
        [5, 900, 6, 910, 5, 905], calls=3, per_call=2
    )
    assert (setup, zero, mm, truncated) == ((), (5, 6, 5), (900, 910, 905), False)


def test_split_takes_the_setup_interval_first():
    setup, (k,), truncated = kd.split_intervals(
        [40, 7, 8], calls=2, per_call=1, setup=1
    )
    assert (setup, k, truncated) == ((40,), (7, 8), False)


def test_a_truncated_stream_keeps_its_labels():
    # The buffer filled after the second call's zero: that is a prefix, so
    # every interval it holds is still at its call's position.
    setup, (zero, mm), truncated = kd.split_intervals([5, 900, 6], calls=3, per_call=2)
    assert (zero, mm, truncated) == ((5, 6), (900,), True)


def test_more_intervals_than_declared_is_an_error():
    with pytest.raises(RuntimeError, match="does not declare"):
        kd.split_intervals([1, 17, 1, 19], calls=2, per_call=1)


def test_the_flush_pairs_after_the_last_call_are_dropped():
    # What npu2 decoded: the core's flush pairs follow the kernel's last call.
    setup, (k,), truncated = kd.split_intervals(
        [3907, 3906, 1, 8, 18], calls=2, per_call=1, flush=3
    )
    assert (k, truncated) == ((3907, 3906), False)


@pytest.mark.parametrize("tail", [[3905], [1, 33, 18], [1, 8, 33]])
def test_a_long_interval_after_the_last_call_is_not_a_flush(tail):
    with pytest.raises(RuntimeError, match="does not declare"):
        kd.split_intervals([3907, 3906, *tail], calls=2, per_call=1, flush=3)


def test_more_flush_intervals_than_pairs_is_an_error():
    with pytest.raises(RuntimeError, match="does not declare"):
        kd.split_intervals([3907, 3906, 1, 1, 1, 18], calls=2, per_call=1, flush=3)


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([17, 19], kd.CallCycles(kernel=(17, 19))),
        ([17], kd.CallCycles(kernel=(17,), truncated=True)),
        ([], RuntimeError),
        # Each call emitted two pairs, longer than a flush pair.
        ([170, 190, 170, 190], RuntimeError),
    ],
)
def test_cycle_protocol_requires_one_complete_pair_per_call(
    monkeypatch, tmp_path, fn, intervals, expected
):
    cfg = SimpleNamespace(
        physical_mlir_path="physical.mlir", trace_to_json=lambda *args: None
    )
    monkeypatch.setattr(kd, "TraceConfig", lambda **kwargs: cfg)
    monkeypatch.setattr(kd, "upload", lambda *args, **kwargs: ([], object()))
    monkeypatch.setattr(kd, "get_cycles_summary", lambda path: [(0, *intervals)])

    def run():
        return kd.cycles_per_call(
            lambda *args, **kwargs: None,
            [],
            1,
            int,
            fn=fn,
            trace_size=1024,
            workdir=tmp_path,
            calls=2,
        )

    if expected is RuntimeError:
        with pytest.raises(RuntimeError, match="trace"):
            run()
    else:
        assert run() == expected
