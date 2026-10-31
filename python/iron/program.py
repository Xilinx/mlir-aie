# program.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import logging

import numpy as np

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects.aie import TraceMode  # pyright: ignore[reportAttributeAccessIssue]
from ..extras.context import mlir_mod_ctx  # pyright: ignore[reportMissingImports]
from ..helpers.errors import design_error
from ..helpers.sourceloc import SourceSite
from .configuration import DeviceConfiguration
from .device import Device
from .runtime import Runtime
from .scratchpad_parameter import ScratchpadParameter

logger = logging.getLogger(__name__)


class Program:
    """One compilation unit and host-visible execution graph.

    A Program emits one MLIR module. It composes one or more
    [`DeviceConfiguration`][aie.iron.configuration.DeviceConfiguration] images and selects the
    entry [`Runtime`][aie.iron.Runtime] that orchestrates one run. Each device
    configuration owns the resources inside one ``aie.device`` operation;
    Program owns module creation, entry selection, and cross-configuration
    validation.

    A Program may have several entries: runtimes the host dispatches by name
    from one compiled build, each taking the same arguments. The first is the
    default entry.
    """

    def __init__(
        self,
        device: Device | None,
        rt: "Runtime | list[Runtime]",
        workers: "list | None" = None,
        *,
        expand_load_pdis: bool | None = None,
        load_pdi_to_ctrl_pkt: bool | None = None,
        reconfig_teardown: bool | None = None,
        reconfig_init_entry: bool | None = None,
    ):
        """Construct a Program with all design information needed to run the design on a device.

        !!! note
            MLIR verification (`ctx.module.operation.verify()`) is performed inside
            [`resolve_program`][aie.iron.program.Program.resolve_program], not during construction.

        Args:
            device (Device): The device used to generate the final MLIR for the design.
                Accepts the ``Device | None`` returned by ``iron.get_current_device``
                directly and raises if no device has been selected, so callers need
                not narrow it first.
            rt (Runtime | list[Runtime]): The design's entry runtime, or its
                entries, the first the default. Entries take identical
                argument types.
            workers (list[Worker] | None, optional): The Workers to run on the
                device. Defaults to None (no workers). Workers are passed here
                explicitly rather than started from within the runtime sequence.
            expand_load_pdis: The value passed to aiecc's
                ``--expand-load-pdis`` option during full-ELF compilation.
                ``None`` omits the option.
            load_pdi_to_ctrl_pkt: The value passed to aiecc's
                ``--load-pdi-to-ctrl-pkt`` option during full-ELF compilation.
                ``None`` omits the option.
            reconfig_teardown: The value passed to aiecc's
                ``--reconfig-teardown`` option. ``None`` omits the option.
            reconfig_init_entry: The value passed to aiecc's
                ``--reconfig-init-entry`` option. ``None`` omits the option.

        Raises:
            ValueError: If ``device`` is None (no NPU device was selected/detected),
                or the entries take different argument types.
        """
        if device is None:
            raise ValueError(
                "Program requires a device, but none was selected. Pass an explicit "
                "Device, or ensure an NPU runtime is available for "
                "iron.get_current_device()."
            )
        entries = list(rt) if isinstance(rt, list) else [rt]
        configuration = DeviceConfiguration(
            "main", device, workers=workers or (), runtimes=entries
        )
        self._configurations = [configuration]
        self._entries = entries
        self._entry = entries[0]
        self._implicit_configuration = True
        self._reconfiguration_options = {
            "expand_load_pdis": expand_load_pdis,
            "load_pdi_to_ctrl_pkt": load_pdi_to_ctrl_pkt,
            "reconfig_teardown": reconfig_teardown,
            "reconfig_init_entry": reconfig_init_entry,
        }
        self._site = SourceSite.capture()
        self._assign_names()

    @classmethod
    def compose(
        cls,
        configurations: "list[DeviceConfiguration]",
        *,
        entry: "Runtime | list[Runtime]",
        expand_load_pdis: bool | None = None,
        load_pdi_to_ctrl_pkt: bool | None = None,
        reconfig_teardown: bool | None = None,
        reconfig_init_entry: bool | None = None,
    ) -> "Program":
        """Compose configurations into one Program run from ``entry``.

        Args:
            configurations: The device configurations, each one ``aie.device``.
            entry: The runtime that orchestrates one run, or the entries the
                host dispatches by name, the first the default. Entries take
                identical argument types.
            expand_load_pdis: The value passed to aiecc's
                ``--expand-load-pdis`` option. ``None`` omits the option.
            load_pdi_to_ctrl_pkt: The value passed to aiecc's
                ``--load-pdi-to-ctrl-pkt`` option. ``None`` omits the option.
            reconfig_teardown: The value passed to aiecc's
                ``--reconfig-teardown`` option. ``None`` omits the option.
            reconfig_init_entry: The value passed to aiecc's
                ``--reconfig-init-entry`` option. ``None`` omits the option.
        """
        program = cls.__new__(cls)
        program._configurations = list(configurations)
        program._entries = list(entry) if isinstance(entry, list) else [entry]
        program._entry = program._entries[0]
        program._implicit_configuration = False
        program._reconfiguration_options = {
            "expand_load_pdis": expand_load_pdis,
            "load_pdi_to_ctrl_pkt": load_pdi_to_ctrl_pkt,
            "reconfig_teardown": reconfig_teardown,
            "reconfig_init_entry": reconfig_init_entry,
        }
        program._site = SourceSite.capture()
        program._assign_names()
        program._validate_composition()
        return program

    def _assign_names(self) -> None:
        def assign(items, base: str) -> None:
            taken = {item.name for item in items if item.name is not None}
            suffix = 0
            for item in items:
                if item.name is not None:
                    continue
                while True:
                    name = base if suffix == 0 else f"{base}_{suffix}"
                    suffix += 1
                    if name not in taken:
                        break
                item._assign_name(name)
                taken.add(name)

        assign(self._configurations, "device")
        assign(
            [
                runtime
                for configuration in self._configurations
                for runtime in configuration.runtimes
            ],
            "sequence",
        )

    def _validate_composition(self) -> None:
        if not self._configurations:
            raise ValueError("Program requires at least one configuration.")
        names = [
            configuration.name
            for configuration in self._configurations
            if configuration.name is not None
        ]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(
                f"Program has duplicate configuration names: {duplicates}."
            )
        for entry in self._entries:
            if entry.configuration not in self._configurations:
                raise ValueError(
                    f"Entry runtime {entry.name!r} does not belong to this Program."
                )
        if len({id(entry) for entry in self._entries}) != len(self._entries):
            raise ValueError("Program lists an entry runtime more than once.")
        for entry in self._entries[1:]:
            if entry.arg_types != self._entry.arg_types:
                raise ValueError(
                    f"Entry runtime {entry.name!r} takes {entry.arg_types}, but "
                    f"{self._entry.name!r} takes {self._entry.arg_types}; a "
                    "Program's entries take identical argument types."
                )

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
    ):
        """Enable hardware tracing for this program.

        Configures the AIE trace units and routes trace packets to DDR via the shim DMA.
        Lives on Program (not Runtime) because it configures both the traced
        Workers' tiles and the Runtime's trace-buffer sequencing.

        Args:
            trace_size (int): Size of the trace buffer in bytes.
            workers (list[Worker] | None, optional): Specific workers to trace. If None,
                all workers with ``trace`` set will be traced. Defaults to None.
            reuse_output_buffer (bool, optional): When False (default), trace
                lowering appends a dedicated trace-buffer argument to the
                runtime_sequence; it lands at the tail so enabling trace never
                perturbs the data arguments' indices. When True, trace data is
                written into the tail of the last output buffer, saving a host
                buffer. Defaults to False.
            coretile_events (list | None, optional): List of up to 8 core tile trace events.
                See [the AIEX dialect reference](../AIEXDialect.md) for available
                events under (type)EventAIE such as CoreEventAIE.
                Defaults to None (uses hardware defaults).
            coremem_events (list | None, optional): List of up to 8 core memory trace events.
                Defaults to None (uses hardware defaults).
            memtile_events (list | None, optional): List of up to 8 mem tile trace events.
                Defaults to None (uses hardware defaults).
            shimtile_events (list | None, optional): List of up to 8 shim tile trace events.
                Defaults to None (uses hardware defaults).
            egress_shim_col (int, optional): Column of the shim tile used to
                egress trace packets to DDR. Defaults to 0.
            core_trace_mode (TraceMode, optional): Trace mode for core tiles.
                Defaults to Event-Time.
        """
        configuration = self._entry.configuration
        assert configuration is not None
        configuration.enable_trace(
            trace_size=trace_size,
            workers=workers,
            reuse_output_buffer=reuse_output_buffer,
            coretile_events=coretile_events,
            coremem_events=coremem_events,
            memtile_events=memtile_events,
            shimtile_events=shimtile_events,
            egress_shim_col=egress_shim_col,
            core_trace_mode=core_trace_mode,
        )

    def resolve_program(
        self, device_name: str = "main", context: ir.Context | None = None
    ) -> ir.Module:
        """Resolve the program components in order to generate MLIR.

        Tiles are emitted as aie.logical_tile ops. The --aie-place-tiles pass
        in the compilation pipeline converts them to aie.tile ops.

        Args:
            device_name: The symbol of the generated ``aie.device``.
            context: The context to build the module in, so a caller combining
                several programs' devices can move them rather than print and
                parse them; a new one when None.

        Returns:
            The module holding the design's ``aie.device``.
        """
        try:
            return self._resolve_program(device_name, context)
        except Exception as exc:
            raise design_error(exc)

    def _resolve_program(self, device_name, context):
        if context is None:
            context = ir.Context()
        with context, ir.Location.unknown():
            loc = self._site.location()

        self._validate_composition()
        with mlir_mod_ctx(context=context, location=loc) as ctx:
            scratchpad_parameters: dict[str, ScratchpadParameter] = {}
            for configuration in self._configurations:
                for worker in configuration.workers:
                    for arg in worker.flat_fn_args:
                        if isinstance(arg, ScratchpadParameter):
                            self._register_scratchpad_parameter(
                                scratchpad_parameters, arg
                            )
                for runtime in configuration.runtimes:
                    for parameter in runtime.scratchpad_parameters:
                        self._register_scratchpad_parameter(
                            scratchpad_parameters, parameter
                        )
                for parameter in configuration.scratchpad_parameters:
                    self._register_scratchpad_parameter(
                        scratchpad_parameters, parameter
                    )
            for parameter in scratchpad_parameters.values():
                parameter.resolve()

            owners = {}
            for configuration in self._configurations:
                assert configuration.name is not None
                symbol = (
                    device_name if self._implicit_configuration else configuration.name
                )
                configuration.resolve(
                    device_name=symbol,
                    loc=loc,
                    entries=self._entries,
                    owners=owners,
                )

            for configuration in self._configurations:
                for runtime in configuration.runtimes:
                    for parameter in runtime.scratchpad_parameters:
                        self._register_scratchpad_parameter(
                            scratchpad_parameters, parameter
                        ).resolve()

            qualified = []
            for entry in self._entries:
                assert entry.configuration is not None and entry.name is not None
                symbol = (
                    device_name
                    if self._implicit_configuration
                    else entry.configuration.name
                )
                qualified.append(f"{symbol}:{entry.name}")
            if not self._implicit_configuration or len(qualified) > 1:
                ctx.module.operation.attributes["iron.entry"] = ir.StringAttr.get(
                    qualified[0]
                )
            if len(qualified) > 1:
                ctx.module.operation.attributes["iron.entries"] = ir.ArrayAttr.get(
                    [ir.StringAttr.get(name) for name in qualified]
                )
            if not self._implicit_configuration:
                ctx.module.operation.attributes["iron.configuration_count"] = (
                    ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), len(self._configurations)
                    )
                )

            for name, value in self._reconfiguration_options.items():
                if value is not None:
                    ctx.module.operation.attributes[f"iron.{name}"] = ir.BoolAttr.get(
                        value
                    )

            self._print_verify(ctx)
            return ctx.module

    @staticmethod
    def _register_scratchpad_parameter(
        parameters: dict[str, ScratchpadParameter],
        parameter: ScratchpadParameter,
    ) -> ScratchpadParameter:
        declaration = parameters.get(parameter.name)
        if declaration is None:
            parameters[parameter.name] = parameter
            return parameter
        # numpy.intc and numpy.int32 are distinct classes on Windows.
        if np.dtype(declaration.dtype) != np.dtype(parameter.dtype):
            raise ValueError(
                f"ScratchpadParameter {parameter.name!r} has conflicting dtypes "
                f"{declaration.dtype} and {parameter.dtype}."
            )
        if declaration.attributes != parameter.attributes:
            raise ValueError(
                f"ScratchpadParameter {parameter.name!r} is declared with "
                f"attributes {declaration.attributes} and "
                f"{parameter.attributes}."
            )
        if parameter is not declaration:
            parameter._resolved = True
        return declaration

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if not verify:
            raise RuntimeError(f"MLIR module failed verification: {verify}")
