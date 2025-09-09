# functional.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""
Functional programming utilities inspired by C++ <functional> header.
Provides function objects for common operations that can be used with for_each and other algorithms.
"""

import operator
from typing import Any, Callable


class UnaryFunction:
    """Base class for unary function objects."""

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError


class BinaryFunction:
    """Base class for binary function objects."""

    def __call__(self, x: Any, y: Any) -> Any:
        raise NotImplementedError


# Unary function objects
class negate(UnaryFunction):
    """Function object for unary minus (negation)."""

    def __call__(self, x: Any) -> Any:
        return -1 * x


class logical_not(UnaryFunction):
    """Function object for logical NOT."""

    def __call__(self, x: Any) -> Any:
        return not x


class bit_not(UnaryFunction):
    """Function object for bitwise NOT."""

    def __call__(self, x: Any) -> Any:
        return ~x


class identity(UnaryFunction):
    """Function object that returns its argument unchanged."""

    def __call__(self, x: Any) -> Any:
        return x


# Binary function objects
class plus(BinaryFunction):
    """Function object for addition."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x + y


class minus(BinaryFunction):
    """Function object for subtraction."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x - y


class multiplies(BinaryFunction):
    """Function object for multiplication."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x * y


class divides(BinaryFunction):
    """Function object for division."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x / y


class modulus(BinaryFunction):
    """Function object for modulo operation."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x % y


class bit_and(BinaryFunction):
    """Function object for bitwise AND."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x & y


class bit_or(BinaryFunction):
    """Function object for bitwise OR."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x | y


class bit_xor(BinaryFunction):
    """Function object for bitwise XOR."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x ^ y


class equal_to(BinaryFunction):
    """Function object for equality comparison."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x == y


class not_equal_to(BinaryFunction):
    """Function object for inequality comparison."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x != y


class greater(BinaryFunction):
    """Function object for greater than comparison."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x > y


class less(BinaryFunction):
    """Function object for less than comparison."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x < y


class greater_equal(BinaryFunction):
    """Function object for greater than or equal comparison."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x >= y


class less_equal(BinaryFunction):
    """Function object for less than or equal comparison."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x <= y


class logical_and(BinaryFunction):
    """Function object for logical AND."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x and y


class logical_or(BinaryFunction):
    """Function object for logical OR."""

    def __call__(self, x: Any, y: Any) -> Any:
        return x or y


negate = negate()
logical_not = logical_not()
bit_not = bit_not()
identity = identity()

plus = plus()
minus = minus()
multiplies = multiplies()
divides = divides()
modulus = modulus()
bit_and = bit_and()
bit_or = bit_or()
bit_xor = bit_xor()
equal_to = equal_to()
not_equal_to = not_equal_to()
greater = greater()
less = less()
greater_equal = greater_equal()
less_equal = less_equal()
logical_and = logical_and()
logical_or = logical_or()
