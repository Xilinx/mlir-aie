# test_declare_operator.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""The declaration half of Operator: construction, copies, layout and keys — no NPU required."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pytest
from aie.helpers.npdtypes import v8bfp16ebs8, v16bfp16ebs16
from aie.iron.device import NPU1Col1, NPU2Col4
from aie.utils import hostruntime
from aie.utils.compile.jit.markers import In, Out
from aie.utils.compile.jit.operator import Operator, Unresolvable
from aie.utils.compile.jit.params import param, per_call, tunable


class Placement(Enum):
    SHIMS = 1
    MEMTILES = 2


class Scale(Operator):
    M: int = param()
    rows: int = param(default=lambda op: op.M * 2)
    pad: int = param(derive=lambda op: op.M + 1, repr=False)
    gated: bool = param(default=False)
    factor: float = param(default=1.0, array=True)
    tile: int | None = tunable()
    placement: Placement = tunable(default=Placement.SHIMS, domain=Placement)
    lanes: int | None = tunable(init=False)
    x = In(M, obj_shape=tile)
    gate = In(M, when=gated)
    y = Out(M)
    count = per_call(derive=lambda op: op.M // 4)

    def resolve(self, dev):
        return self.replace(tile=self.tile or 16, lanes=dev.cols)


DEV = NPU2Col4()


def test_an_operator_sets_its_parameters_given_defaulted_or_computed():
    op = Scale(M=8)
    assert (op.M, op.rows, op.pad, op.gated, op.tile, op.lanes) == (
        8,
        16,
        9,
        False,
        None,
        None,
    )
    assert op.placement is Placement.SHIMS
    assert Scale(M=8, rows=3).rows == 3


def test_an_operator_binds_its_present_members():
    op = Scale(M=8)
    assert op.y.shape == (8,) and op.count.op is op
    with pytest.raises(AttributeError, match="declared when=gated, which is False"):
        op.gate
    assert Scale(M=8, gated=True).gate.shape == (8,)


@pytest.mark.parametrize(
    "given, error, message",
    [
        ({}, TypeError, "Scale\\(\\) needs \\['M'\\]"),
        ({"M": 8, "K": 1}, TypeError, "takes no argument \\['K'\\]"),
        ({"M": 8, "lanes": 1}, TypeError, "takes no argument \\['lanes'\\]"),
        ({"M": 8, "pad": 3}, ValueError, "pad \\(3\\) must be 9, what its other"),
    ],
)
def test_a_construction_breaking_the_declaration_is_refused(given, error, message):
    with pytest.raises(error, match=message):
        Scale(**given)


class Computed(Operator):
    M: int = param(default=lambda op: None)


class Nullable(Operator):
    M: int | None = param(default=lambda op: None)


def test_a_computed_default_of_none_must_be_given_unless_nullable():
    with pytest.raises(
        ValueError, match="M \\(None\\) must be given: nothing computes it"
    ):
        Computed()
    assert Nullable().M is None


def test_a_copy_recomputes_a_derive_and_keeps_every_other_value():
    op = Scale(M=8).replace(M=4)
    assert (op.M, op.rows, op.pad) == (4, 16, 5)
    assert Scale(M=8).replace(lanes=2).lanes == 2
    with pytest.raises(TypeError, match="Scale has no parameter \\['K'\\]"):
        Scale(M=8).replace(K=1)


def test_resolving_fills_every_tunable_on_a_copy():
    op = Scale(M=8)
    new = op.resolved(DEV)
    assert (new.tile, new.lanes) == (16, 4) and new.resolved(DEV) is new


def test_an_operator_resolved_for_one_device_resolves_again_for_another():
    four = Scale(M=8).resolved(DEV)
    one = four.resolved(NPU1Col1())
    assert (four.lanes, one.lanes) == (4, 1)
    assert four.resolved(NPU2Col4()) is four


class Sized(Scale):
    def resolve(self, dev):
        return self.replace(tile=self.tile or 4 * dev.cols, lanes=dev.cols)


def test_a_copy_of_a_resolved_operator_resolves_again_from_the_unresolved_one():
    four = Sized(M=8).resolved(DEV)
    copy = four.replace()
    assert (copy.tile, copy.lanes) == (16, 4)
    one = copy.resolved(NPU1Col1())
    assert (
        (one.tile, one.lanes)
        == (4, 1)
        == (
            four.resolved(NPU1Col1()).tile,
            four.resolved(NPU1Col1()).lanes,
        )
    )
    assert four.replace(M=4).resolved(NPU1Col1()).M == 4


def test_two_resolutions_for_different_devices_have_different_keys():
    four = Scale(M=8).resolved(DEV)
    one = Scale(M=8).resolved(NPU1Col1()).replace(lanes=4)
    assert (one.tile, one.lanes) == (four.tile, four.lanes)
    assert one.array_key() != four.array_key()
    assert one.design_key() != four.design_key()


def test_a_copy_keeps_its_resolution():
    four = Scale(M=8).resolved(DEV)
    copy = four.copy()
    assert copy is not four and copy.resolved(DEV) is copy
    assert copy.resolved(NPU1Col1()).lanes == 1
    assert Scale(M=8).copy().resolved(DEV).lanes == 4


class Lazy(Scale):
    def resolve(self, dev):
        return self.replace()


class Same(Scale):
    def resolve(self, dev):
        return self


class Other(Scale):
    def resolve(self, dev):
        return Scale(M=self.M).replace(tile=16, lanes=4)


def test_a_resolve_that_leaves_a_tunable_or_returns_itself_is_refused():
    with pytest.raises(Unresolvable, match="left \\['tile', 'lanes'\\] unset"):
        Lazy(M=8).resolved(DEV)
    with pytest.raises(TypeError, match="must return a copy"):
        Same(M=8).resolved(DEV)
    with pytest.raises(TypeError, match="must return a copy"):
        Other(M=8).resolved(DEV)


def test_the_layout_holds_the_key_and_names_any_other_read():
    op = Scale(M=8).resolved(DEV)
    view = op.layout()
    assert (view.tile, view.placement, view.factor, view.lanes) == (
        16,
        Placement.SHIMS,
        1.0,
        4,
    )
    assert view.x.op is view
    for read in (lambda: view.M, lambda: view.x.shape):
        with pytest.raises(
            AttributeError,
            match="Scale.build\\(\\) reads M, which is not in the core-layout key",
        ):
            read()


def test_the_keys_are_the_layout_for_cores_and_every_parameter_for_a_build():
    op = Scale(M=8).resolved(DEV)
    assert op.array_key() == (
        Scale,
        ("device", "npu2_4col"),
        ("factor", 1.0),
        ("tile", 16),
        ("placement", 1),
        ("lanes", 4),
    )
    assert op.replace(M=4).array_key() == op.array_key()
    assert op.replace(M=4).design_key() != op.design_key()
    traced = Scale(M=8, trace="t").resolved(DEV)
    assert traced.array_key()[-1] == ("trace", "t")
    assert traced.design_key()[1][:2] == (("device", "npu2_4col"), ("trace", "t"))


class Typed(Operator):
    M: int = param()
    dtype: object = param(default=np.float32, array=True)
    x = In(M)


def test_a_key_holds_one_spelling_of_each_value():
    spellings = [
        (Typed(M=64), Typed(M=np.int64(64))),
        (Typed(M=8), Typed(M=8, dtype=np.dtype(np.float32))),
    ]
    for a, b in spellings:
        for key in (Operator.array_key, Operator.design_key):
            assert key(a) == key(b) and hash(key(a)) == hash(key(b))
            assert repr(key(a)) == repr(key(b))


def test_a_block_float_type_keys_as_its_class():
    a, b = Typed(M=8, dtype=v8bfp16ebs8), Typed(M=8, dtype=v8bfp16ebs8)
    wider = Typed(M=8, dtype=v16bfp16ebs16)
    assert ("dtype", v8bfp16ebs8) in a.array_key()
    assert ("dtype", v8bfp16ebs8) in a.design_key()[1]
    for key in (Operator.array_key, Operator.design_key):
        assert key(a) == key(b) and hash(key(a)) == hash(key(b))
        assert key(a) != key(wider)


def test_an_operator_prints_its_shown_parameters():
    assert repr(Scale(M=8)) == (
        "Scale(M=8, rows=16, gated=False, factor=1.0, tile=None, "
        "placement=<Placement.SHIMS: 1>, lanes=None)"
    )


class Library(Operator, library=True):
    def sequence(self):
        pass


class Reference(Library):
    reference_only = True
    n: int = param()
    x = In(n, dtype=np.float32)


def test_a_library_base_is_not_declared_and_a_reference_only_operator_has_no_kernel():
    assert "declaration" not in vars(Library)
    assert Reference(n=2).x.dtype is np.float32
    with pytest.raises(TypeError, match="Hiding.sequence hides Library.sequence"):

        class Hiding(Library):
            sequence: int = param()

    with pytest.raises(
        TypeError, match="Kernel is reference_only, so it declares no build"
    ):

        class Kernel(Library):
            reference_only = True

            def build(self):
                pass


class Ordered(Operator):
    total: int = param(default=lambda op: op.M + op.tile)
    M: int = param(default=3)
    tile: int = tunable(default=4)


def test_a_computed_default_sees_every_plain_default():
    assert Ordered().total == 7


def test_an_enum_tunable_takes_only_its_enum():
    with pytest.raises(TypeError, match="placement \\(1\\) must be a Placement"):
        Scale(M=8, placement=1)


class Checked(Scale):
    def compatible(self):
        if self.tile > self.M:
            raise ValueError("tile exceeds M")


def test_a_resolved_copy_is_checked_against_its_tunables():
    with pytest.raises(ValueError, match="tile exceeds M"):
        Checked(M=8, tile=32).resolved(DEV)


class OnDevice(Scale):
    def compatible(self):
        assert hostruntime._CURRENT_DEVICE is DEV


def test_a_resolved_copy_is_checked_with_its_device_current():
    OnDevice(M=8).resolved(DEV)
