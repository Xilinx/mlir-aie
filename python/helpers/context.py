import contextlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

from .. import ir


@dataclass
class MLIRContext:
    context: ir.Context
    module: ir.Module

    def __str__(self):
        return str(self.module)


@contextmanager
def mlir_mod_ctx(
    src: str | None = None,
    context: ir.Context | None = None,
    location: ir.Location | None = None,
    allow_unregistered_dialects=False,
) -> MLIRContext:
    if context is None:
        context = ir.Context()
    if allow_unregistered_dialects:
        context.allow_unregistered_dialects = True
    with ExitStack() as stack:
        stack.enter_context(context)
        if location is None:
            location = ir.Location.unknown()
        stack.enter_context(location)
        if src is not None:
            module = ir.Module.parse(src)
        else:
            module = ir.Module.create()
        ip = ir.InsertionPoint(module.body)
        stack.enter_context(ip)
        yield MLIRContext(context, module)
    context._clear_live_operations()


class RAIIMLIRContext:
    context: ir.Context
    location: ir.Location

    def __init__(self, location: ir.Location | None = None):
        self.context = ir.Context()
        self.context.__enter__()
        if location is None:
            location = ir.Location.unknown()
        self.location = location
        self.location.__enter__()

    def __del__(self):
        self.location.__exit__(None, None, None)
        self.context.__exit__(None, None, None)
        # i guess the extension gets destroyed before this object sometimes?
        if ir is not None:
            assert ir.Context is not self.context


class ExplicitlyManagedModule:
    module: ir.Module
    _ip: ir.InsertionPoint

    def __init__(self):
        self.module = ir.Module.create()
        self._ip = ir.InsertionPoint(self.module.body)
        self._ip.__enter__()

    def finish(self):
        self._ip.__exit__(None, None, None)
        return self.module

    def __str__(self):
        return str(self.module)


@contextlib.contextmanager
def enable_multithreading(context=None):
    from ..ir import Context

    if context is None:
        context = Context.current
    context.enable_multithreading(True)
    yield
    context.enable_multithreading(False)


@contextlib.contextmanager
def disable_multithreading(context=None):
    from ..ir import Context

    if context is None:
        context = Context.current

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)
