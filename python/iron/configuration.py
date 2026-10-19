# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import itertools
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects.aie import (
    AIEDevice,  # pyright: ignore[reportAttributeAccessIssue]
    DeviceOp,  # pyright: ignore[reportAttributeAccessIssue]
    RuntimeSequenceOp,  # pyright: ignore[reportAttributeAccessIssue]
    TraceMode,  # pyright: ignore[reportAttributeAccessIssue]
    device,
)
from ..dialects.aiex import (
    ConfigureOp,  # pyright: ignore[reportAttributeAccessIssue]
    ScratchpadParameterOp,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..helpers.dialects.func import FuncBase
from ..helpers.util import mlir_type_to_np_dtype
from ..utils import trace as trace_utils
from ..utils.compile.jit.context import get_compile_arg
from .dataflow.endpoint import ObjectFifoEndpoint
from .dataflow.flow import PacketFlow
from .dataflow.objectfifo import ObjectFifoLink
from .device import Device, NamedDevice
from .kernel import Kernel
from .resolvable import DeviceResources, PerDeviceConfiguration, Resolvable
from .runtime import Runtime
from .runtime._context import active_configuration, active_configuration_scope
from .scratchpad_parameter import ScratchpadParameter


class DeviceConfiguration(DeviceResources):
    """A device image and the runtime sequences that use it."""

    def __init__(
        self,
        name: str | Device | None = None,
        device: Device | None = None,
        *,
        workers: Sequence = (),
        runtimes: Sequence[Runtime] = (),
    ) -> None:
        super().__init__()
        if isinstance(name, Device):
            if device is not None:
                raise TypeError("DeviceConfiguration received two devices.")
            device = name
            name = None
        if device is None:
            raise TypeError("DeviceConfiguration requires a device.")
        if name == "":
            raise ValueError("DeviceConfiguration name must not be empty.")
        self._name = name
        self._device = device
        self._workers = list(workers)
        self._runtimes = list(runtimes)
        self._trace_size = None
        self._trace_workers = None
        self._reuse_output_buffer = False
        self._egress_shim_col = 0
        self._coretile_events = None
        self._coremem_events = None
        self._memtile_events = None
        self._shimtile_events = None
        self._core_trace_mode = TraceMode.EventTime
        self._owners = None
        self._prebuilt: str | None = None
        self._scratchpad_parameters: list[ScratchpadParameter] = []

        runtime_names = [
            runtime.name for runtime in self._runtimes if runtime.name is not None
        ]
        duplicates = sorted(
            name for name in set(runtime_names) if runtime_names.count(name) > 1
        )
        if duplicates:
            raise ValueError(
                f"DeviceConfiguration {self._name!r} has duplicate runtime names: {duplicates}."
            )
        for runtime in self._runtimes:
            runtime._bind_configuration(self)

    @classmethod
    def from_module(cls, name: str, module: ir.Module) -> DeviceConfiguration:
        """Return a configuration that emits ``module``'s one ``aie.device`` verbatim.

        Each runtime sequence in the device becomes a Runtime of the same name
        and argument types. The ``aiex.scratchpad_parameter`` declarations
        beside the device are declared again with all their attributes, so
        ``Program.compose`` takes it like any other.

        Args:
            name: The configuration's name; the device is renamed to it.
            module: The module holding the device and its declarations.

        Returns:
            The configuration.

        Raises:
            ValueError: ``module`` holds no ``aie.device``, or several.
            TypeError: A sequence argument has no NumPy dtype or a dynamic
                shape, or a scratchpad parameter has no NumPy dtype.
        """
        devices = [op for op in module.body.operations if isinstance(op, DeviceOp)]
        if len(devices) != 1:
            raise ValueError(
                f"DeviceConfiguration.from_module needs a module holding one "
                f"aie.device, not {len(devices)}."
            )
        (device_op,) = devices
        with module.context:
            declarations = [
                op
                for op in module.body.operations
                if isinstance(op, ScratchpadParameterOp)
            ]
            parameters = [
                ScratchpadParameter.from_declaration(op) for op in declarations
            ]
            runtimes = []
            for sequence in device_op.body_region.blocks[0].operations:
                if not isinstance(sequence, RuntimeSequenceOp):
                    continue
                types = []
                for arg in sequence.body.blocks[0].arguments:
                    if not isinstance(arg.type, ir.MemRefType):
                        dtype = mlir_type_to_np_dtype(arg.type)
                        if dtype is None:
                            raise TypeError(f"No NumPy dtype for argument {arg.type}.")
                        types.append(dtype)
                        continue
                    memref = ir.MemRefType(arg.type)
                    if not memref.has_static_shape:
                        raise TypeError(
                            f"Argument {memref} has a dynamic shape; a Runtime "
                            "argument's shape is fixed."
                        )
                    dtype = mlir_type_to_np_dtype(memref.element_type)
                    if dtype is None:
                        raise TypeError(f"No NumPy dtype for the elements of {memref}.")
                    types.append(np.ndarray[tuple(memref.shape), np.dtype[dtype]])
                runtime = Runtime(
                    lambda *_: None,
                    types,
                    name=ir.StringAttr(sequence.attributes["sym_name"]).value,
                )
                for parameter in parameters:
                    runtime.register_parameter(parameter)
                runtimes.append(runtime)
            kind = AIEDevice(ir.IntegerAttr(device_op.attributes["device"]).value)
            # The device's reads verify against the declarations, so the text
            # parsed again at resolution keeps them; only the device is cloned.
            text = "\n".join([*(str(op) for op in declarations), str(device_op)])
        device_type = next(
            c for c in NamedDevice.__subclasses__() if c.aie_device == kind
        )
        configuration = cls(name, device_type(), runtimes=runtimes)
        configuration._prebuilt = text
        configuration._scratchpad_parameters = parameters
        return configuration

    # Device-local resources may belong to only one configuration.
    def _claim(self, resource, owners: dict[int, "DeviceConfiguration"]) -> None:
        if isinstance(resource, (Kernel, ScratchpadParameter)):
            return
        if isinstance(resource, PerDeviceConfiguration):
            resource._bind_device_configuration(self)
            return
        owner = owners.get(id(resource))
        if owner is not None and owner is not self:
            raise ValueError(
                f"{type(resource).__name__} already belongs to device configuration "
                f"{owner.name!r}; it cannot also belong to {self.name!r}."
            )
        owners[id(resource)] = self

    def claim_runtime_resource(self, resource) -> None:
        if self._owners is None:
            raise RuntimeError(
                "DeviceConfiguration cannot claim runtime resources before resolution."
            )
        self._claim(resource, self._owners)
        self._claim(resource.tile, self._owners)

    def _claim_fifo_graph(self, handle, owners, visited=None) -> None:
        if visited is None:
            visited = set()
        fifo = handle._object_fifo
        if id(fifo) in visited:
            return
        visited.add(id(fifo))
        self._claim(fifo, owners)
        runtime_handles = {
            runtime_handle
            for runtime in self._runtimes
            for runtime_handle in runtime.fifos
        }
        handles = [candidate for candidate in [fifo._prod, *fifo._cons] if candidate]
        for candidate in handles:
            self._claim(candidate, owners)
            endpoint = candidate.endpoint
            if endpoint in self._workers:
                pass
            elif candidate in runtime_handles:
                pass
            elif isinstance(endpoint, ObjectFifoEndpoint) and not isinstance(
                endpoint, PerDeviceConfiguration
            ):
                pass
            elif not isinstance(endpoint, ObjectFifoLink):
                raise ValueError(
                    f"ObjectFifo {fifo.name!r} has an endpoint outside device "
                    f"configuration {self.name!r}."
                )
            if isinstance(endpoint, PerDeviceConfiguration):
                self._claim(endpoint, owners)
            if isinstance(endpoint, ObjectFifoLink):
                self._claim(endpoint, owners)
                for linked in [*endpoint._srcs, *endpoint._dsts]:
                    if linked._object_fifo is not fifo:
                        self._claim_fifo_graph(linked, owners, visited)

    @property
    def name(self) -> str | None:
        return self._name

    def _assign_name(self, name: str) -> None:
        if self._name is None:
            self._name = name

    @property
    def device(self) -> Device:
        return self._device

    @property
    def workers(self) -> list:
        return list(self._workers)

    @property
    def runtimes(self) -> list[Runtime]:
        return list(self._runtimes)

    @property
    def scratchpad_parameters(self) -> list[ScratchpadParameter]:
        """The parameters a prebuilt device declares beside it, used or not by a sequence."""
        return list(self._scratchpad_parameters)

    @contextmanager
    def configure(self):
        if active_configuration() is not None:
            raise RuntimeError("Nested configuration scopes are not supported.")
        op = ConfigureOp(self._name)
        block = ir.Block.create_at_start(op.body)
        with ir.InsertionPoint(block), active_configuration_scope(self):
            yield self

    def enable_trace(
        self,
        trace_size: int | None = None,
        workers: list | None = None,
        reuse_output_buffer: bool = False,
        coretile_events: list | None = None,
        coremem_events: list | None = None,
        memtile_events: list | None = None,
        shimtile_events: list | None = None,
        egress_shim_col: int = 0,
        core_trace_mode=TraceMode.EventTime,
    ) -> None:
        self._trace_size = trace_size
        self._trace_workers = workers
        self._reuse_output_buffer = reuse_output_buffer
        self._coretile_events = coretile_events
        self._coremem_events = coremem_events
        self._memtile_events = memtile_events
        self._shimtile_events = shimtile_events
        self._core_trace_mode = core_trace_mode
        self._egress_shim_col = egress_shim_col

    # Device-local symbols share one namespace inside this configuration: an
    # explicit name may be claimed once, and unnamed ones take fresh names.
    def _name_unnamed(self) -> None:
        fifos = [h._object_fifo for runtime in self._runtimes for h in runtime.fifos]
        fifos += [h._object_fifo for worker in self._workers for h in worker.fifos]
        fifos = list(dict.fromkeys(fifos))
        for fifo in fifos:
            for handle in [fifo._prod, *fifo._cons]:
                link = handle.endpoint if handle is not None else None
                if isinstance(link, ObjectFifoLink):
                    reached = [h._object_fifo for h in [*link._srcs, *link._dsts]]
                    fifos += [
                        reached_fifo
                        for reached_fifo in reached
                        if reached_fifo not in fifos
                    ]
        rtps = [
            buffer
            for worker in self._workers
            for buffer in worker.buffers
            if buffer._use_write_rtp
        ]
        rtps = list(dict.fromkeys(rtps))
        named = [(fifo.name, "ObjectFifo") for fifo in fifos if fifo.name is not None]
        named += [(b._name, "Buffer") for b in rtps if b._name is not None]
        claimed = {}
        for name, kind in named:
            if name in claimed:
                kinds = (
                    f"two {kind}s"
                    if claimed[name] == kind
                    else "an ObjectFifo and a Buffer"
                )
                raise ValueError(
                    f"configuration {self._name!r} names symbol {name!r} twice "
                    f"({kinds}); leave one unnamed or rename it"
                )
            claimed[name] = kind
        taken = set(claimed)
        taken.update(flow.name for flow in self._flows if flow.name is not None)

        def fresh(prefix):
            name = next(
                candidate
                for index in itertools.count()
                if (candidate := f"{prefix}{index}") not in taken
            )
            taken.add(name)
            return name

        for fifo in fifos:
            if fifo.name is None:
                fifo.name = fresh("of")
        for buffer in rtps:
            if buffer._name is None:
                buffer._name = fresh("rtp")
        for flow in self._flows:
            if flow.name is None:
                flow._name = fresh(
                    "packetflow" if isinstance(flow, PacketFlow) else "flow"
                )

    def resolve(
        self,
        *,
        device_name: str,
        loc: ir.Location,
        entry: Runtime,
        owners: dict[int, "DeviceConfiguration"],
    ) -> None:
        """Emit this configuration as one ``aie.device`` operation."""
        # The remaining emission order matches the dependencies between device ops.
        self._owners = owners
        if self._prebuilt is not None:
            prebuilt = ir.Module.parse(self._prebuilt)  # alive while cloned
            (op,) = [o for o in prebuilt.body.operations if isinstance(o, DeviceOp)]
            clone = op.clone(ip=ir.InsertionPoint.current)
            old = ir.StringAttr(clone.attributes["sym_name"]).value
            clone.attributes["sym_name"] = ir.StringAttr.get(device_name)
            ir.SymbolTable.replace_all_symbol_uses(old, device_name, clone.operation)
            if entry in self._runtimes and get_compile_arg("_iron_full_elf"):
                (sequence,) = [
                    o
                    for o in clone.body_region.blocks[0].operations
                    if isinstance(o, RuntimeSequenceOp)
                    and ir.StringAttr(o.attributes["sym_name"]).value == entry.name
                ]
                entry.load_pdi_first(sequence.body.blocks[0], device_name)
            return
        self._name_unnamed()
        device_type = type(self._device)
        self._device = device_type()  # pyright: ignore[reportCallIssue]
        current_device = self._device

        for runtime in self._runtimes:
            self._claim(runtime, owners)
        for worker in self._workers:
            self._claim(worker, owners)

        @device(current_device.resolve(), sym_name=device_name, loc=loc)
        def device_body():
            all_fifos = set()
            for runtime in self._runtimes:
                all_fifos.update(runtime.fifos)
            for worker in self._workers:
                all_fifos.update(worker.fifos)
            all_fifos = sorted(all_fifos, key=lambda obj: obj.name)

            all_tiles = []
            for worker in self._workers:
                all_tiles.append(worker.tile)
                for barrier in worker._barriers:
                    self._claim(barrier, owners)
                for arg in worker.flat_fn_args:
                    if isinstance(arg, Resolvable):
                        self._claim(arg, owners)
                        all_tiles.extend(arg.tiles())
            for handle in all_fifos:
                self._claim_fifo_graph(handle, owners)
                all_tiles.extend(
                    [endpoint.tile for endpoint in handle.all_of_endpoints()]
                )
                delegate = handle._object_fifo._delegate_tile
                if delegate is not None:
                    all_tiles.append(delegate)
            for flow in self._flows:
                self._claim(flow, owners)
                all_tiles.extend(flow.all_tiles())
            for tile_dma in self._tile_dmas:
                self._claim(tile_dma, owners)
                all_tiles.extend(tile_dma.all_tiles())
                buffers, locks = tile_dma.all_buffers_and_locks()
                for resource in [*buffers, *locks]:
                    self._claim(resource, owners)
            for lock in self._locks:
                self._claim(lock, owners)
                all_tiles.append(lock.tile)

            for tile in all_tiles:
                self._claim(tile, owners)
                current_device.resolve_tile(tile)

            for handle in all_fifos:
                handle.resolve()
            for lock in self._locks:
                lock.resolve()
            for tile_dma in self._tile_dmas:
                buffers, locks = tile_dma.all_buffers_and_locks()
                for lock in locks:
                    lock.resolve()
                for buffer in buffers:
                    buffer.place(tile_dma.tile)
                    buffer.resolve()

            for worker in self._workers:
                for arg in worker.flat_fn_args:
                    if isinstance(arg, FuncBase):
                        arg.emit()
                    elif isinstance(arg, Resolvable):
                        if arg not in self._flows and arg not in self._tile_dmas:
                            arg.resolve()

            for worker in self._workers:
                worker.resolve()
            for worker in self._workers:
                for cascade in worker._outgoing_cascades:
                    if cascade._dst not in self._workers:
                        raise ValueError(
                            "CascadeFlow endpoints must belong to the same device "
                            f"configuration {self.name!r}."
                        )
                    self._claim(cascade, owners)
                    self._claim(cascade._dst, owners)
                    cascade.resolve()

            self.resolve_tile_dmas()

            if self._trace_workers is not None:
                tiles_to_trace = [worker.tile.op for worker in self._trace_workers]
            else:
                tiles_to_trace = [
                    worker.tile.op
                    for worker in self._workers
                    if worker.trace is not None
                ]
            if self._trace_size is not None and self._trace_size > 0:
                trace_utils.configure_trace(
                    tiles_to_trace,
                    coretile_events=self._coretile_events,
                    coremem_events=self._coremem_events,
                    memtile_events=self._memtile_events,
                    shimtile_events=self._shimtile_events,
                    core_trace_mode=self._core_trace_mode,
                )

            for runtime in self._runtimes:
                implicit_configure_device_ref = (
                    device_name
                    if runtime is entry and get_compile_arg("_iron_full_elf")
                    else None
                )
                runtime.resolve(
                    trace_size=self._trace_size,
                    reuse_output_buffer=self._reuse_output_buffer,
                    egress_shim_col=self._egress_shim_col,
                    implicit_configure_device_ref=implicit_configure_device_ref,
                    device=current_device,
                )

            for flow in self._flows:
                flow.resolve()
