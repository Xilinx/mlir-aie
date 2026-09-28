# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from util import construct_test

# RUN: %python %s | FileCheck %s

# A pattern is right when it walks the elements numpy's own indexing selects.
# Checking it against numpy on REAL arrays is the point: an earlier
# implementation derived the answer from a pointer difference into a zero-byte
# allocation, which segfaulted on an all-integer key and returned a plausible
# but fabricated offset for advanced indexing. Both went unnoticed because
# nothing compared the result to the elements it claims to describe.

SHAPES = [(16,), (4, 3), (16, 16, 512), (2, 3, 4, 5)]

KEYS = [
    np.s_[...],
    np.s_[:],
    np.s_[0],
    np.s_[-1],
    np.s_[1:],
    np.s_[:2],
    np.s_[::2],
    np.s_[1::2],
    np.s_[0::2, 1::2, ...],
    np.s_[..., 0],
    np.s_[0, ...],
    np.s_[None, ...],
    np.s_[:, None],
    np.s_[1, 2],
    np.s_[:, 0],
    np.s_[-2:],
    np.s_[100:],
    np.s_[-100:100:3],
    np.s_[..., None, -1],
    np.s_[None, ..., None],
    (np.int64(-1),),
]


def walked(tap):
    """Return the flat indices the pattern visits, in order."""
    return [
        sum(i * s for i, s in zip(idx, tap.strides)) + tap.offset
        for idx in itertools.product(*(range(s) for s in tap.sizes))
    ]


# CHECK-LABEL: from_slice_matches_numpy
@construct_test
def from_slice_matches_numpy():
    checked = 0
    for shape in SHAPES:
        flat = np.arange(int(np.prod(shape))).reshape(shape)
        for key in KEYS:
            try:
                selected = flat[key]
            except IndexError:
                try:
                    TensorAccessPattern.from_slice(shape, key)
                    raise AssertionError(f"{shape} {key}: expected IndexError")
                except IndexError:
                    pass
                continue
            if np.size(selected) == 0:
                try:
                    TensorAccessPattern.from_slice(shape, key)
                    raise AssertionError(f"{shape} {key}: empty slice accepted")
                except ValueError:
                    pass
                continue
            tap = TensorAccessPattern.from_slice(shape, key)
            assert walked(tap) == list(
                np.asarray(selected).reshape(-1)
            ), f"{shape} {key}: {tap} walks the wrong elements"
            checked += 1
    assert checked > 50, f"only {checked} cases exercised"
    print(f"matched numpy on {checked} (shape, key) pairs")


# CHECK: matched numpy on {{[0-9]+}} (shape, key) pairs


# CHECK-LABEL: from_slice_rejects_what_it_cannot_walk
@construct_test
def from_slice_rejects_what_it_cannot_walk():
    # Advanced indexing selects elements no strided walk reaches.
    for key in (
        np.array([0, 2]),
        [0, 2],
        np.array([True, False] * 8),
        True,
        False,
        np.bool_(True),
        np.bool_(False),
    ):
        try:
            TensorAccessPattern.from_slice((16,), key)
            raise AssertionError(f"{key!r} should be rejected")
        except TypeError:
            pass

    # A buffer descriptor steps forward only.
    for key in (np.s_[::-1], np.s_[::0], np.s_[3:3]):
        try:
            TensorAccessPattern.from_slice((16,), key)
            raise AssertionError(f"{key!r} should be rejected")
        except ValueError:
            pass

    # Malformed keys report the way numpy reports them.
    for shape, key in (((4, 3), np.s_[0, 0, 0]), ((4, 3), 9), ((4, 3), (..., ...))):
        try:
            TensorAccessPattern.from_slice(shape, key)
            raise AssertionError(f"{key!r} should be rejected")
        except IndexError:
            pass
    print("rejected every key a strided walk cannot express")


# CHECK: rejected every key a strided walk cannot express


# CHECK-LABEL: from_slice_all_integer_key
@construct_test
def from_slice_all_integer_key():
    # Names one element, so it is a one-element walk -- not zero dimensions,
    # and not a read through a stand-in array's backing storage.
    tap = TensorAccessPattern.from_slice((1024, 1024, 1024), (1023, 1023, 1023))
    assert tap.sizes == [1] and tap.strides == [1]
    assert tap.offset == 1023 * 1024 * 1024 + 1023 * 1024 + 1023
    print(f"all-integer key -> offset {tap.offset}, sizes {tap.sizes}")


# CHECK: all-integer key -> offset 1073741823, sizes [1]
