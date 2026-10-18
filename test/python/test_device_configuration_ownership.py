# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

import re

import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.iron import (
    Bd,
    Buffer,
    CascadeFlow,
    DeviceConfiguration,
    DmaChannel,
    DmaEndpoint,
    Flow,
    Kernel,
    Lock,
    ObjectFifo,
    PacketFlow,
    PerDeviceConfigurationResolvable,
    Program,
    Resolvable,
    Runtime,
    ScratchpadParameter,
    TileDma,
    Worker,
    WorkerRuntimeBarrier,
    sync_parameters,
)
from aie.iron.device import NPU2Col1, Tile
from aie.ir import Context, Module, StringAttr

Tensor = np.ndarray[(4,), np.dtype[np.int32]]


def runtime(name):
    return Runtime(lambda: None, [], name=name)


def resolve_two(workers_a=(), workers_b=(), configure=None):
    rt_a = runtime("seq_a")
    rt_b = runtime("seq_b")
    config_a = DeviceConfiguration(
        "dev_a", NPU2Col1(), workers=workers_a, runtimes=[rt_a]
    )
    config_b = DeviceConfiguration(
        "dev_b", NPU2Col1(), workers=workers_b, runtimes=[rt_b]
    )
    if configure is not None:
        configure(config_a, config_b, rt_a, rt_b)
    return Program.compose([config_a, config_b], entry=rt_a).resolve_program()


def worker_with(arg, tile=None):
    return Worker(lambda _arg: None, [arg], tile=tile, while_true=False)


class DeviceEmitter(PerDeviceConfigurationResolvable):
    def __init__(self, tile):
        self.tile = tile

    def tiles(self):
        return [self.tile]

    def resolve(self, loc=None, ip=None):
        pass


class StructuralDeviceEmitter:
    def __init__(self, tile):
        self.tile = tile

    def tiles(self):
        return [self.tile]

    def resolve(self, loc=None, ip=None):
        pass


assert isinstance(StructuralDeviceEmitter(None), Resolvable)


@pytest.mark.parametrize(
    "resource_factory",
    [
        lambda tile: Buffer(Tensor, tile=tile, name="shared"),
        lambda tile: Lock(tile, name="shared"),
        lambda tile: DeviceEmitter(tile),
        lambda tile: StructuralDeviceEmitter(tile),
    ],
    ids=["buffer", "lock", "owned-resolvable", "structural-resolvable"],
)
def test_worker_resource_cannot_span_device_configurations(resource_factory):
    tile = Tile(0, 2, tile_type=AIETileType.CoreTile)
    resource = resource_factory(tile)
    worker_a = worker_with(resource, Tile(0, 3, tile_type=AIETileType.CoreTile))
    worker_b = worker_with(resource, Tile(0, 4, tile_type=AIETileType.CoreTile))

    with pytest.raises(ValueError, match="already belongs to device configuration"):
        resolve_two([worker_a], [worker_b])


@pytest.mark.parametrize("flow_type", [Flow, PacketFlow], ids=["flow", "packet-flow"])
def test_flow_cannot_span_device_configurations(flow_type):
    source = Tile(0, 1, tile_type=AIETileType.MemTile)
    destination = Tile(0, 2, tile_type=AIETileType.CoreTile)
    flow = (
        flow_type(source, destination)
        if flow_type is Flow
        else flow_type(0, source, destination)
    )

    def configure(config_a, config_b, rt_a, rt_b):
        rt_a.add_flow(flow)
        rt_b.add_flow(flow)

    with pytest.raises(ValueError, match="already belongs to device configuration"):
        resolve_two(configure=configure)


def test_tile_dma_graph_cannot_span_device_configurations():
    tile = Tile(0, 1, tile_type=AIETileType.MemTile)
    buffer = Buffer(Tensor, tile=tile, name="shared")
    lock = Lock(tile, name="shared")
    channel = DmaChannel(
        DMAChannelDir.MM2S,
        0,
        [Bd(buffer, acquires=[], releases=[])],
    )
    tile_dma = TileDma(tile, [channel])

    def configure(config_a, config_b, rt_a, rt_b):
        for runtime_instance in (rt_a, rt_b):
            runtime_instance.add_lock(lock)
            runtime_instance.add_tile_dma(tile_dma)

    with pytest.raises(ValueError, match="already belongs to device configuration"):
        resolve_two(configure=configure)


def test_runtime_task_resource_cannot_span_device_configurations():
    tile = Tile(0, 1, tile_type=AIETileType.MemTile)
    buffer = Buffer(Tensor, tile=tile, name="shared")

    def configure(config_a, config_b, rt_a, rt_b):
        def task_body():
            DmaEndpoint(tile, DMAChannelDir.MM2S, 0).task(buffer).start().free()

        rt_a._seq_fn = task_body
        rt_b._seq_fn = task_body

    with pytest.raises(ValueError, match="already belongs to device configuration"):
        resolve_two(configure=configure)


def test_object_fifo_cannot_connect_two_device_configurations():
    fifo = ObjectFifo(Tensor, name="shared")
    producer = Worker(lambda handle: None, [fifo.prod()], while_true=False)
    consumer = Worker(lambda handle: None, [fifo.cons()], while_true=False)

    with pytest.raises(ValueError, match="ObjectFifo.*endpoint outside"):
        resolve_two([producer], [consumer])


def test_object_fifo_link_cannot_span_device_configurations():
    source = ObjectFifo(Tensor, name="source")
    linked = source.cons().forward(name="linked")
    producer = Worker(lambda handle: None, [source.prod()], while_true=False)
    consumer = Worker(lambda handle: None, [linked.cons()], while_true=False)

    with pytest.raises(ValueError, match="ObjectFifo.*endpoint outside"):
        resolve_two([producer], [consumer])


def test_cascade_flow_cannot_span_device_configurations():
    producer = Worker(lambda: None, [], while_true=False)
    consumer = Worker(lambda: None, [], while_true=False)
    CascadeFlow(producer, consumer)

    with pytest.raises(ValueError, match="CascadeFlow endpoints must belong"):
        resolve_two([producer], [consumer])


def test_barrier_cannot_span_device_configurations():
    barrier = WorkerRuntimeBarrier()
    worker_a = worker_with(barrier)
    worker_b = worker_with(barrier)

    with pytest.raises(ValueError, match="already belongs to device configuration"):
        resolve_two([worker_a], [worker_b])


def test_module_scoped_resolvables_can_span_device_configurations():
    kernel = Kernel("shared_kernel", "kernel.o", arg_types=[])
    parameter = ScratchpadParameter("shared_parameter", np.int32)

    def body(shared_kernel, shared_parameter):
        shared_kernel()
        shared_parameter.read()

    worker_a = Worker(body, [kernel, parameter], while_true=False)
    worker_b = Worker(body, [kernel, parameter], while_true=False)
    module = resolve_two([worker_a], [worker_b])
    text = str(module)

    assert text.count("func.func private @shared_kernel") == 2
    assert text.count("aiex.scratchpad_parameter @shared_parameter") == 1


def test_distinct_scratchpad_parameters_with_one_name_share_a_declaration():
    parameter_a = ScratchpadParameter("shared_parameter", np.int32)
    parameter_b = ScratchpadParameter("shared_parameter", np.int32)
    worker_a = worker_with(parameter_a)
    worker_b = worker_with(parameter_b)

    text = str(resolve_two([worker_a], [worker_b]))

    assert text.count("aiex.scratchpad_parameter @shared_parameter") == 1


def test_distinct_scratchpad_parameters_reject_conflicting_dtypes():
    parameter_a = ScratchpadParameter("shared_parameter", np.int32)
    parameter_b = ScratchpadParameter("shared_parameter", np.int16)
    worker_a = worker_with(parameter_a)
    worker_b = worker_with(parameter_b)

    with pytest.raises(ValueError, match="conflicting dtypes"):
        resolve_two([worker_a], [worker_b])


def test_device_descriptor_can_define_multiple_configurations():
    device = NPU2Col1()
    rt_a = runtime("seq_a")
    rt_b = runtime("seq_b")
    config_a = DeviceConfiguration("dev_a", device, runtimes=[rt_a])
    config_b = DeviceConfiguration("dev_b", device, runtimes=[rt_b])

    module = Program.compose([config_a, config_b], entry=rt_a).resolve_program()

    assert str(module).count("aie.device(npu2_1col)") == 2


def sequence_local_configuration(name):
    fifo = ObjectFifo(Tensor, name=f"fifo_{name}")
    barrier = WorkerRuntimeBarrier()
    worker = Worker(
        lambda handle, ready: None,
        [fifo.cons(), barrier],
        while_true=False,
    )

    def sequence(host, handle):
        sync_parameters()
        barrier.set(1)
        handle.fill(host)

    runtime_instance = Runtime(
        sequence,
        [Tensor, fifo.prod()],
        name=f"seq_{name}",
    )
    return (
        DeviceConfiguration(
            f"dev_{name}",
            NPU2Col1(),
            workers=[worker],
            runtimes=[runtime_instance],
        ),
        runtime_instance,
    )


def test_sequence_local_resolvables_are_distinct_per_configuration():
    config_a, runtime_a = sequence_local_configuration("a")
    config_b, _ = sequence_local_configuration("b")
    module = Program.compose([config_a, config_b], entry=runtime_a).resolve_program()
    text = str(module)

    assert text.count("aiex.sync_scratchpad_parameters_from_host") == 2
    assert text.count("aiex.set_lock") == 2
    assert text.count("aiex.dma_configure_task_for") == 2


def test_distinct_device_resources_resolve_in_distinct_configurations():
    worker_a = worker_with(Buffer(Tensor, name="local"))
    worker_b = worker_with(Buffer(Tensor, name="local"))
    module = resolve_two([worker_a], [worker_b])

    assert str(module).count('sym_name = "local"') == 2


@pytest.mark.parametrize(
    "resource_factory, expected",
    [
        (
            lambda index: Lock(
                Tile(0, 2, tile_type=AIETileType.CoreTile),
                name=f"lock_{index}",
            ),
            "aie.lock",
        ),
        (
            lambda index: DeviceEmitter(Tile(0, 2, tile_type=AIETileType.CoreTile)),
            "aie.device",
        ),
    ],
    ids=["lock", "structural-resolvable"],
)
def test_distinct_worker_resources_can_belong_to_distinct_configurations(
    resource_factory, expected
):
    worker_a = worker_with(resource_factory("a"))
    worker_b = worker_with(resource_factory("b"))

    assert str(resolve_two([worker_a], [worker_b])).count(expected) >= 2


@pytest.mark.parametrize("flow_type", [Flow, PacketFlow], ids=["flow", "packet-flow"])
def test_distinct_flows_can_belong_to_distinct_configurations(flow_type):
    flows = []
    for index in range(2):
        source = Tile(0, 1, tile_type=AIETileType.MemTile)
        destination = Tile(0, 2, tile_type=AIETileType.CoreTile)
        flows.append(
            flow_type(source, destination, name=f"flow_{index}")
            if flow_type is Flow
            else flow_type(index, source, destination, name=f"flow_{index}")
        )

    def configure(config_a, config_b, rt_a, rt_b):
        rt_a.add_flow(flows[0])
        rt_b.add_flow(flows[1])

    op_name = "aie.route from" if flow_type is Flow else "aie.packet_flow("
    assert str(resolve_two(configure=configure)).count(op_name) == 2


def make_tile_dma(name):
    tile = Tile(0, 1, tile_type=AIETileType.MemTile)
    buffer = Buffer(Tensor, tile=tile, name=f"buffer_{name}")
    lock = Lock(tile, name=f"lock_{name}")
    return lock, TileDma(
        tile,
        [DmaChannel(DMAChannelDir.MM2S, 0, [Bd(buffer)])],
    )


def test_distinct_tile_dma_graphs_can_belong_to_distinct_configurations():
    lock_a, tile_dma_a = make_tile_dma("a")
    lock_b, tile_dma_b = make_tile_dma("b")

    def configure(config_a, config_b, rt_a, rt_b):
        rt_a.add_lock(lock_a)
        rt_a.add_tile_dma(tile_dma_a)
        rt_b.add_lock(lock_b)
        rt_b.add_tile_dma(tile_dma_b)

    assert str(resolve_two(configure=configure)).count("aie.memtile_dma") == 2


def fifo_worker_pair(name):
    fifo = ObjectFifo(Tensor, name=name)
    producer = Worker(lambda handle: None, [fifo.prod()], while_true=False)
    consumer = Worker(lambda handle: None, [fifo.cons()], while_true=False)
    return producer, consumer


def linked_fifo_worker_pair(name):
    source = ObjectFifo(Tensor, name=f"{name}_source")
    linked = source.cons().forward(name=f"{name}_linked")
    producer = Worker(lambda handle: None, [source.prod()], while_true=False)
    consumer = Worker(lambda handle: None, [linked.cons()], while_true=False)
    return producer, consumer


@pytest.mark.parametrize(
    "pair_factory, fifo_count",
    [(fifo_worker_pair, 2), (linked_fifo_worker_pair, 4)],
    ids=["object-fifo", "object-fifo-link"],
)
def test_distinct_fifo_graphs_can_belong_to_distinct_configurations(
    pair_factory, fifo_count
):
    workers_a = pair_factory("a")
    workers_b = pair_factory("b")
    text = str(resolve_two(workers_a, workers_b))

    assert text.count("aie.objectfifo @") == fifo_count


def test_distinct_cascade_flows_can_belong_to_distinct_configurations():
    producer_a = Worker(lambda: None, [], while_true=False)
    consumer_a = Worker(lambda: None, [], while_true=False)
    producer_b = Worker(lambda: None, [], while_true=False)
    consumer_b = Worker(lambda: None, [], while_true=False)
    CascadeFlow(producer_a, consumer_a)
    CascadeFlow(producer_b, consumer_b)

    text = str(resolve_two([producer_a, consumer_a], [producer_b, consumer_b]))

    assert text.count("aie.cascade_flow") == 2


Line = np.ndarray[(256,), np.dtype[np.int32]]


def member(fifo_name=None, rtp_name=None):
    """A Worker draining a fifo its Runtime fills, with an RTP buffer."""
    of = ObjectFifo(Line, name=fifo_name)
    rtp = Buffer(Tensor, name=rtp_name, use_write_rtp=True)

    def core(c, _rtp):
        c.acquire(1)
        c.release(1)

    def sequence(a, p):
        p.fill(a)

    return Worker(core, [of.cons(), rtp]), Runtime(sequence, [Line, of.prod()])


def pack(*members):
    workers, runtimes = zip(*members)
    config = DeviceConfiguration("pack", NPU2Col1(), workers=workers, runtimes=runtimes)
    return Program.compose([config], entry=runtimes[0]).resolve_program()


@pytest.mark.parametrize(
    "first,second,kinds",
    [
        (dict(fifo_name="in"), dict(fifo_name="in"), "two ObjectFifos"),
        (dict(rtp_name="rtp"), dict(rtp_name="rtp"), "two Buffers"),
        (dict(fifo_name="x"), dict(rtp_name="x"), "an ObjectFifo and a Buffer"),
    ],
)
def test_one_configuration_refuses_a_symbol_named_twice(first, second, kinds):
    # Two members built by one function name their fifos alike; the verifier
    # used to report the second's consumer port on the first's tile.
    name = first.get("fifo_name") or first["rtp_name"]
    with pytest.raises(
        ValueError,
        match=rf"configuration 'pack' names symbol '{name}' twice \({kinds}\); "
        "leave one unnamed or rename it",
    ):
        pack(member(**first), member(**second))


def test_unnamed_symbols_in_one_configuration_take_fresh_names():
    module = pack(member(), member(fifo_name="of0", rtp_name="rtp0"))
    module.operation.verify()
    text = str(module)
    for symbol in (
        "aie.objectfifo @of0(",
        "aie.objectfifo @of1(",
        'sym_name = "rtp0"',
        'sym_name = "rtp1"',
    ):
        assert text.count(symbol) == 1, symbol


def test_two_configurations_may_name_a_symbol_alike():
    (w_a, rt_a), (w_b, rt_b) = member(fifo_name="in"), member(fifo_name="in")
    config_a = DeviceConfiguration("dev_a", NPU2Col1(), workers=[w_a], runtimes=[rt_a])
    config_b = DeviceConfiguration("dev_b", NPU2Col1(), workers=[w_b], runtimes=[rt_b])
    Program.compose(
        [config_a, config_b], entry=rt_a
    ).resolve_program().operation.verify()


def shim_dma(rt):
    buffer = Buffer(Tensor, tile=Tile(0, 1, tile_type=AIETileType.MemTile))
    channel = DmaChannel(DMAChannelDir.MM2S, 0, [Bd(buffer)])
    rt.add_tile_dma(TileDma(Tile(0, 0), [channel]))


def control_route():
    source = Tile(0, 1, tile_type=AIETileType.MemTile)
    return PacketFlow(3, source, Tile(0, 0), dst_port=WireBundle.TileControl)


def program(*registers):
    """One configuration of one Runtime per ``registers`` entry, which adds to it."""
    runtimes = [runtime(f"seq_{i}") for i in range(len(registers))]
    for rt, register in zip(runtimes, registers):
        register(rt)
    config = DeviceConfiguration("pack", NPU2Col1(), runtimes=runtimes)
    return Program.compose([config], entry=runtimes[0]).resolve_program()


@pytest.mark.parametrize(
    "second, kinds",
    [
        (shim_dma, "a TileDma and a TileDma"),
        (lambda rt: rt.add_flow(control_route()), "a TileDma and a PacketFlow"),
    ],
)
def test_co_resident_runtimes_cannot_program_one_tile_dma(second, kinds):
    with pytest.raises(
        ValueError,
        match=r"Runtimes 'seq_0' and 'seq_1' both program the DMA of tile "
        rf"\(0, 0\) in device configuration 'pack', by {kinds}\.",
    ):
        program(shim_dma, second)


def test_co_resident_runtimes_may_share_one_tile_program():
    route = control_route()
    module = program(lambda rt: rt.add_flow(route), lambda rt: rt.add_flow(route))
    assert str(module).count("aie.packet_flow(") == 1


PREBUILT = """
aiex.scratchpad_parameter @count : i32
aie.device(npu2_1col) @made_elsewhere {
  %t = aie.tile(0, 2)
  %c = aie.core(%t) {
    aie.end
  }
  aie.runtime_sequence @run(%a: memref<16xi32>, %b: memref<4x8xbf16>) {
  }
}
"""


def test_a_prebuilt_device_composes_verbatim():
    with Context():
        module = Module.parse(PREBUILT)
        prebuilt = DeviceConfiguration.from_module("other", module)
        module.body.operations[1].attributes["sym_name"] = StringAttr.get("other")
        device = str(module.body.operations[1])
    (run,) = prebuilt.runtimes
    assert run.name == "run" and [p.name for p in run.scratchpad_parameters] == [
        "count"
    ]
    rt = runtime("seq")
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[rt])
    composed = Program.compose([built, prebuilt], entry=rt).resolve_program()
    assert str(composed.body.operations[2]) == device
    assert str(composed).count("aiex.scratchpad_parameter @count") == 1


def test_a_prebuilt_runtime_can_be_the_entry():
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(PREBUILT))
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[runtime("seq")])
    (run,) = prebuilt.runtimes
    composed = Program.compose([built, prebuilt], entry=run).resolve_program()
    assert StringAttr(composed.operation.attributes["iron.entry"]).value == "other:run"
    assert str(composed).count("aiex.scratchpad_parameter @count") == 1


@pytest.mark.parametrize(
    "text", ["", PREBUILT + PREBUILT.split("\n", 2)[2].replace("made_", "built_")]
)
def test_a_prebuilt_module_holds_one_device(text):
    with Context(), pytest.raises(ValueError, match="holding one aie.device, not"):
        DeviceConfiguration.from_module("other", Module.parse(text))


def test_a_runtime_sharing_one_of_two_tile_programs_conflicts_with_the_other():
    # seq_0 claims the shared TileDma before its own route.
    shared = []

    def own_and_share(rt):
        shim_dma(rt)
        shared.extend(rt.tile_dmas)
        rt.add_flow(control_route())

    with pytest.raises(ValueError, match="both program the DMA of tile"):
        program(own_and_share, lambda rt: rt.add_tile_dma(shared[0]))


def test_merged_runtimes_issue_one_sequence():
    (w_a, rt_a), (w_b, rt_b) = member(), member()
    flow = Flow(Tile(0, 0), Tile(0, 1, tile_type=AIETileType.MemTile), src_channel=0)
    rt_c = Runtime(lambda a: flow.fill(a), [Line])
    rt_c.add_flow(flow)
    merged = Runtime.merge([rt_a, rt_b, rt_c], name="steps")
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[w_a, w_b], runtimes=[merged]
    )
    module = Program.compose([config], entry=merged).resolve_program()
    text = str(module)
    assert text.count("aie.runtime_sequence") == 1
    assert "@steps(%arg0: memref<256xi32>, %arg1: memref<256xi32>, " in text
    assert text.count("aiex.dma_start_task") == 3


def test_a_merged_step_waits_for_the_last_steps_transfers():
    # A drains into the host buffer B then fills from; one step's await comes first.
    of_in, of_out = ObjectFifo(Line, name="into"), ObjectFifo(Line, name="out_of")
    worker = Worker(lambda a, b: None, [of_in.cons(), of_out.prod()])
    rt_a = Runtime(lambda c, h: h.drain(c, wait=True), [Line, of_out.cons()])
    rt_b = Runtime(lambda c, h: h.fill(c), [Line, of_in.prod()])
    merged = Runtime.merge([rt_a, rt_b])
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[worker], runtimes=[merged]
    )
    text = str(Program.compose([config], entry=merged).resolve_program())
    sequence = text[text.index("aie.runtime_sequence") :]
    steps = re.findall(r"aiex\.dma_(start|await|free)_task", sequence)
    assert steps == ["start", "await", "free", "start", "free"]


def test_a_configured_runtime_does_not_merge():
    rt = runtime("seq")
    DeviceConfiguration("dev", NPU2Col1(), runtimes=[rt])
    with pytest.raises(ValueError, match="merge it before configuring"):
        Runtime.merge([rt, runtime("other")])
