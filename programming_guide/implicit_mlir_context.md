<!---//===- implicit_mlir_context.md ----------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# The Implicit MLIR Context

IRON designs look like ordinary Python functions, but the body of a
function decorated with [`@iron.jit`](../python/utils/jit.py) does not
*return* an MLIR module the way a normal builder would.  Instead, the
body executes inside an implicit MLIR context — a thread-local
`Location` and `InsertionPoint` managed by `aie.extras.context` — and
each `Lock`, `Buffer`, `ObjectFifo`, kernel call, and other primitive
*mutates* that context as a side effect.

This short page explains the model.  Once you have it, several IRON
patterns that otherwise look like magic become straightforward, and
the error messages that mention "no active location" become
self-explanatory.

## What the user writes vs. what happens

A minimal `@iron.jit` design:

```python
@iron.jit
def passthrough(a_in: In, b_out: Out):
    of = ObjectFifo(line_ty, name="in")
    of_out = ObjectFifo(line_ty, name="out")

    def core_fn(of_in, of_out, kernel):
        elem_in  = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_out, LINE_SIZE)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [of.cons(), of_out.prod(), kernel])

    rt = Runtime()
    with rt.sequence(line_ty, line_ty) as (a, b):
        rt.start(worker)
        rt.fill(of.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()
```

When `passthrough` is called, IRON wraps the body in an implicit
`mlir_mod_ctx()`.  Every `ObjectFifo(...)`, `Worker(...)`, `rt.fill(...)`,
and `kernel(...)` call that runs inside that body emits MLIR
operations into the active `InsertionPoint` of that context.  The
final `Program.resolve_program()` call walks the user-level Python
objects (Workers, ObjectFifos, Runtime tasks) and emits the remaining
ops that close out the module.

The "return value" of the body is the assembled MLIR module — but the
work of building it happens through context-mutating side effects, not
through a value passed up the call stack.

## Why no operation needs a context argument

Because the context is thread-local, primitives like:

```python
elem = of_in.acquire(1)
kernel(elem_in, elem_out)
Lock(tile).acquire(value=1)
```

read the active `Location` and `InsertionPoint` from `aie.extras.context`
when they need them.  This is what lets the design read like normal
Python: there is no `ctx` parameter to thread through every call site.

The trade-off is that these calls only make sense *inside* an
`@iron.jit` body (or another active MLIR context).  Calling
`Lock(tile).acquire(value=1)` from a module-level script will raise a
"no active location" error.

## Consequence: `@func` pykernels must live at module scope

A pykernel is a Python function decorated with
[`@func`](../python/helpers/dialects/func.py) whose body becomes an
AIE compute-core function.  `@func` resolves its argument types when
the decorator runs, which requires an active MLIR Location at
decoration time.

The robust pattern is to keep the `@func` declaration at module top
level so it inherits the import-time context, and to close over the
shape/dtype constants used in its signature:

```python
VECTOR_SIZE = 4096
LINE_SIZE = VECTOR_SIZE // 4
_LINE_TY = np.ndarray[(LINE_SIZE,), np.dtype[np.uint8]]

@func
def passthrough_fn(input: _LINE_TY, output: _LINE_TY, line_width: np.int32):
    for i in range_(line_width):
        output[i] = input[i]

@iron.jit
def passthrough_pykernel(a_in: In, b_out: Out):
    ...
    worker = Worker(core_fn, [..., passthrough_fn])
    ...
```

See
[`programming_examples/basic/passthrough_pykernel/passthrough_pykernel.py`](../programming_examples/basic/passthrough_pykernel/passthrough_pykernel.py)
for the full design.

Because parameter types are baked at import time, `_LINE_TY` cannot
depend on a runtime CLI flag — the build-time `VECTOR_SIZE` constant is
the single source of truth.

## Consequence: re-resolving a Program re-creates its Device

If you call `Program.resolve_program()` more than once (for example
during interactive notebook iteration), the Device object is
re-instantiated each time.  This is intentional: the first resolve
attached MLIR operations to the Device's tile objects, and those
operations belong to the previous (now-closed) MLIR module.  Re-using
the same Device instance would leave stale ops dangling.

The mechanism is internal to
[`python/iron/program.py`](../python/iron/program.py); the practical
takeaway is that each `resolve_program()` call produces a fresh,
independent MLIR module.

## Reading "no active location" errors

When you see:

```
RuntimeError: no active location
```

or a similar message naming `InsertionPoint`, the proximate cause is
almost always that an IRON primitive (`Buffer(...)`, `Lock(...)`,
`of.acquire(...)`, a kernel call, ...) ran *outside* an `@iron.jit`
body.  Common triggers:

* Constructing IR objects in a module-level helper that is not invoked
  from inside the JIT body.
* Decorating a `@func` pykernel inside another function instead of at
  module scope.
* Calling the `@iron.jit`-decorated design from inside another design's
  body (instead of from host-side Python).

The fix is always the same: move the construction into the `@iron.jit`
body, or hoist the `@func` definition to module level.

## What stays explicit

Several IRON objects are ordinary Python: their constructors do *not*
touch the implicit context.  They are registered with it later, when
`Program.resolve_program()` walks the design.

* `Worker(core_fn, fn_args, tile=...)`
* `Runtime()`
* `Program(device, rt)`
* `ObjectFifo(obj_type, depth, name=...)`

You can create, store, and pass these around freely outside any
context.  Only their `.resolve()` methods (called from
`Program.resolve_program()`) emit MLIR.

## Summary

* `@iron.jit` runs the function body inside an implicit
  `mlir_mod_ctx()` with a thread-local `Location` and
  `InsertionPoint`.
* IRON primitives read that thread-local state at construction or
  call time, which is why no `ctx` parameter is threaded through.
* The implicit-context model is what makes `@func` need to live at
  module scope, what motivates the per-resolve Device re-creation,
  and what produces "no active location" errors when a primitive runs
  outside the JIT body.
* `Worker`, `Runtime`, `Program`, `ObjectFifo` constructors are pure
  Python; only `Program.resolve_program()` emits the module.
