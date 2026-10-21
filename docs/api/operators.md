<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Operators and declarations (`aie.utils.compile.jit`)

An operator class declares what it takes in its body: its parameters
(`param`), the choices a device resolves (`tunable`), its operands (`In`,
`Out`, `InOut`) and the values each call supplies (`per_call`). The class
body is read into a `Declaration` when it finishes, so the shapes, the cache
keys and the per-call words all come from one place. A library's operator
base (IRON's) subclasses `Operator` with `library=True` and adds building,
tuning and dispatch.

```python
from aie.utils.compile.jit.markers import In, Out
from aie.utils.compile.jit.operator import Operator
from aie.utils.compile.jit.params import param, per_call, tunable


class Scale(Operator):
    M: int = param()
    factor: float = param(default=1.0)
    tile: int | None = tunable()
    x = In(M, obj_shape=tile)
    y = Out(M)
    count = per_call(derive=lambda op: op.M // 4)

    def resolve(self, dev):
        return self.replace(tile=self.tile or 16)


op = Scale(M=1024)          # Scale(M=1024, factor=1.0, tile=None)
done = op.resolved(device)  # tile=16; done.is_resolved, done.resolved_for
```

A per-call value is an expression in one integer the host gives each call
(`PerCall` and `Affine` in `aie.helpers.taplib`). The host evaluates it, and
a pattern that needs it on the device stages it as `arith`.

## Operator

::: utils.compile.jit.operator.Operator
    options:
      show_root_heading: true
      heading_level: 3

::: utils.compile.jit.operator.Unresolvable
    options:
      show_root_heading: true
      heading_level: 3

## Parameters and per-call values

::: utils.compile.jit.params
    options:
      show_root_heading: false
      members:
        - OperatorParam
        - param
        - tunable
        - per_call
        - Select

## Operands

::: utils.compile.jit.markers
    options:
      show_root_heading: false
      members:
        - In
        - Out
        - InOut

## Declaration

::: utils.compile.jit.declaration.Declaration
    options:
      show_root_heading: true
      heading_level: 3

## Per-call expressions

::: helpers.taplib.PerCall
    options:
      show_root_heading: true
      heading_level: 3

::: helpers.taplib.Affine
    options:
      show_root_heading: true
      heading_level: 3
