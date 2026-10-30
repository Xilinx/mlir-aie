# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

import re

import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.dialects.aie import DeviceOp
from aie.ir import Context, MLIRError, Module, StringAttr
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
    Port,
    Program,
    Resolvable,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    TileDma,
    Worker,
    WorkerRuntimeBarrier,
    sync_parameters,
)
from aie.iron.controlflow import if_
from aie.iron.device import NPU2, NPU2Col1, Tile
from aie.passmanager import PassManager
from aie.utils.compile.jit.context import compile_context

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
    """Return a Worker draining a fifo its Runtime fills, with an RTP buffer."""
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


def mem_tile():
    return Tile(0, 1, tile_type=AIETileType.MemTile)


def static_program(name, tile, direction, channel, loop=True):
    """A Runtime that adds a TileDma running one channel of ``tile``."""
    rt = runtime(name)
    bd = Bd(Buffer(Tensor, tile=tile))
    rt.add_tile_dma(TileDma(tile, [DmaChannel(direction, channel, [bd], loop=loop)]))
    return rt


def task_on(name, tile, direction, channel):
    """A Runtime whose sequence runs one task on a channel of ``tile``."""
    buffer = Buffer(Tensor, tile=tile)
    endpoint = DmaEndpoint(tile, direction, channel)
    return Runtime(lambda: endpoint.task(Bd(buffer)).start().free(), [], name=name)


def control_route(tile):
    return PacketFlow(3, Tile(0, 0), tile, dst_port=WireBundle.TileControl)


def program(*runtimes):
    config = DeviceConfiguration("pack", NPU2Col1(), runtimes=runtimes)
    return Program.compose([config], entry=runtimes[0]).resolve_program()


def compile_tasks(module):
    """Place, give channels and assign runtime BD ids, as aiecc does."""
    PassManager.parse(
        "builtin.module(aie.device(aie-place-tiles,aie-objectfifo-allocate,"
        "aie-assign-runtime-sequence-bd-ids))",
        context=module.context,
    ).run(module.operation)


def test_co_resident_runtimes_run_static_programs_on_one_tile():
    mem = mem_tile()
    text = str(
        program(
            static_program("seq_0", mem, DMAChannelDir.MM2S, 0),
            static_program("seq_1", mem, DMAChannelDir.S2MM, 0),
        )
    )
    assert text.count("aie.memtile_dma(") == 1
    assert "aie.dma_start(MM2S, 0" in text and "aie.dma_start(S2MM, 0" in text


def test_one_channel_in_two_static_programs_is_refused():
    mem = mem_tile()
    with pytest.raises(ValueError, match="already has MM2S channel 0"):
        program(
            static_program("seq_0", mem, DMAChannelDir.MM2S, 0),
            static_program("seq_1", mem, DMAChannelDir.MM2S, 0),
        )


@pytest.mark.parametrize("same_object", [True, False])
@pytest.mark.parametrize("one_runtime", [True, False])
def test_a_control_route_into_a_tile_with_a_static_program_is_refused(
    one_runtime, same_object
):
    mem = mem_tile()
    static = static_program("seq_0", mem, DMAChannelDir.MM2S, 0)
    routed = static if one_runtime else runtime("seq_1")
    routed.add_flow(control_route(mem if same_object else mem_tile()))
    with pytest.raises(
        ValueError,
        match=r"PacketFlow 'packetflow0' writes the TileControl of "
        r"Tile<MemTile>\(0, 1\), which runs a TileDma program",
    ):
        program(*dict.fromkeys([static, routed]))


def test_co_resident_runtimes_may_share_one_control_route():
    route = control_route(mem_tile())
    first, second = runtime("seq_0"), runtime("seq_1")
    first.add_flow(route)
    second.add_flow(route)
    assert str(program(first, second)).count("aie.packet_flow(") == 1


def test_a_task_on_a_channel_a_looping_static_program_runs_fails():
    mem = mem_tile()
    module = program(
        static_program("seq_0", mem, DMAChannelDir.MM2S, 0),
        task_on("seq_1", mem, DMAChannelDir.MM2S, 0),
    )
    with pytest.raises(MLIRError, match="static DMA program loops forever"):
        compile_tasks(module)


def gathered(mem):
    """Two packet routes into ``mem`` on one channel the compiler assigns."""
    owner = PacketFlow(1, Tile(0, 2), mem, src_channel=None, dst_channel=None)
    sharer = PacketFlow(
        2, Tile(0, 3), mem, src_channel=None, dst_channel=owner.endpoint(mem)
    )
    return owner, sharer


def static_on(name, tile, channel, flows=()):
    """A Runtime with ``flows`` and a TileDma running S2MM on ``channel``."""
    rt = runtime(name)
    for flow in flows:
        rt.add_flow(flow)
    bd = Bd(Buffer(Tensor, tile=tile))
    rt.add_tile_dma(TileDma(tile, [DmaChannel(DMAChannelDir.S2MM, channel, [bd])]))
    return rt


def test_a_route_sharing_a_channel_gives_the_owners_endpoint():
    mem = mem_tile()
    owner, sharer = gathered(mem)
    assert sharer.endpoint(mem) is owner.endpoint(mem)
    with pytest.raises(ValueError, match="already has S2MM channel"):
        TileDma(
            mem,
            [
                DmaChannel(
                    DMAChannelDir.S2MM,
                    owner.endpoint(mem),
                    [Bd(Buffer(Tensor, tile=mem))],
                ),
                DmaChannel(
                    DMAChannelDir.S2MM,
                    sharer.endpoint(mem),
                    [Bd(Buffer(Tensor, tile=mem))],
                ),
            ],
        )


def test_a_task_through_a_sharing_route_on_a_looping_static_channel_fails():
    mem = mem_tile()
    owner, sharer = gathered(mem)
    static = static_on("seq_0", mem, owner.endpoint(mem), (owner, sharer))
    task = Runtime(
        lambda: sharer.endpoint(mem).task(Bd(Buffer(Tensor, tile=mem))).start().free(),
        [],
        name="seq_1",
    )
    with pytest.raises(MLIRError, match="static DMA program loops forever"):
        compile_tasks(program(static, task))


def test_the_compiler_keeps_a_static_channel_off_an_indexed_task():
    mem = mem_tile()
    owner, sharer = gathered(mem)
    static = static_on("seq_0", mem, owner.endpoint(mem), (owner, sharer))
    module = program(static, task_on("seq_1", mem, DMAChannelDir.S2MM, 0))
    compile_tasks(module)
    assert "aie.dma_start(S2MM, 0" not in str(module)


def test_a_task_may_follow_a_static_program_that_ends():
    mem = mem_tile()
    compile_tasks(
        program(
            static_program("seq_0", mem, DMAChannelDir.MM2S, 0, loop=False),
            task_on("seq_1", mem, DMAChannelDir.MM2S, 0),
        )
    )


def test_tiles_pinned_to_one_column_alone_are_two_tiles():
    first = Tile(col=0, tile_type=AIETileType.MemTile)
    second = Tile(col=0, tile_type=AIETileType.MemTile)
    module = program(
        static_program("seq_0", first, DMAChannelDir.MM2S, 0),
        task_on("seq_1", second, DMAChannelDir.MM2S, 0),
    )
    assert str(module).count("aiex.dma_configure_task(") == 1


def test_tasks_may_share_a_channel_and_leave_a_static_one_alone():
    mem = mem_tile()
    module = program(
        static_program("seq_0", mem, DMAChannelDir.MM2S, 0),
        task_on("seq_1", mem, DMAChannelDir.MM2S, 1),
        task_on("seq_2", mem, DMAChannelDir.MM2S, 1),
    )
    assert str(module).count("aiex.dma_configure_task(") == 2


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


SELF_NAMED = """
aie.device(npu2_1col) @main {
  %t = aie.tile(0, 2)
  %s = aie.tile(0, 0)
  aie.shim_dma_allocation @main (%s, MM2S, 0)
  aie.runtime_sequence @run(%a: memref<16xi32>) {
    aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @main} : memref<16xi32>
  }
}
"""

BUFFER_NAMED = """
aie.device(npu2_1col) @main {
  %t = aie.tile(0, 2)
  %b = aie.buffer(%t) {sym_name = "main"} : memref<16xi32>
  aie.runtime_sequence @run(%a: memref<16xi32>) {
    %c = arith.constant 1 : i32
    aiex.npu.rtp_write(@main, 0, %c) : i32
  }
}
"""


@pytest.mark.parametrize("text", [SELF_NAMED, BUFFER_NAMED], ids=["symbol", "buffer"])
def test_a_prebuilt_device_is_not_renamed_over_a_name_inside_it(text):
    with Context():
        module = Module.parse(text)
        with pytest.raises(ValueError, match="holds an op named @main"):
            DeviceConfiguration.from_module("other", module)
        (run,) = DeviceConfiguration.from_module("main", module).runtimes
        assert run.name == "run"


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


def test_a_prebuilt_full_elf_entry_loads_its_own_pdi():
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(PREBUILT))
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[runtime("seq")])
    (run,) = prebuilt.runtimes
    with compile_context(_iron_full_elf=True):
        composed = Program.compose([built, prebuilt], entry=run).resolve_program()
    (device,) = [
        op
        for op in composed.body.operations
        if isinstance(op, DeviceOp) and StringAttr(op.sym_name).value == "other"
    ]
    assert "aiex.npu.load_pdi {device_ref = @other}" in str(device)
    assert composed.operation.verify()


def test_a_prebuilt_runtime_takes_scalar_arguments():
    text = PREBUILT.replace("%b: memref<4x8xbf16>", "%n: i32")
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(text))
    (run,) = prebuilt.runtimes
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[runtime("seq")])
    composed = Program.compose([built, prebuilt], entry=run).resolve_program()
    assert "aie.runtime_sequence @run(%arg0: memref<16xi32>, %arg1: i32)" in str(
        composed
    )


def test_a_renamed_prebuilt_device_refers_to_its_new_name():
    text = """
aie.device(npu2_1col) @main {
  %t = aie.tile(0, 2)
  aie.runtime_sequence @run(%a: memref<16xi32>) {
    aiex.npu.load_pdi {device_ref = @main}
  }
}
"""
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(text))
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[runtime("seq")])
    (run,) = prebuilt.runtimes
    composed = str(Program.compose([built, prebuilt], entry=run).resolve_program())
    assert "aie.device(npu2_1col) @other" in composed
    assert "device_ref = @other" in composed and "@main" not in composed


def test_a_renamed_prebuilt_device_renames_a_configure_of_itself():
    text = """
aie.device(npu2_1col) @main {
  %t = aie.tile(0, 2)
  aie.runtime_sequence @run(%a: memref<16xi32>) {
    aiex.configure @main {
    }
  }
}
"""
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(text))
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[runtime("seq")])
    (run,) = prebuilt.runtimes
    composed = str(Program.compose([built, prebuilt], entry=run).resolve_program())
    assert "aiex.configure @other" in composed and "@main" not in composed


def test_a_lowered_prebuilt_declaration_keeps_its_attributes():
    attributes = (
        "{kind = 2 : i32, max_value = 7 : i32, min_value = 0 : i32, "
        "state_table_idx = 0 : ui8}"
    )
    text = PREBUILT.replace(
        "aiex.scratchpad_parameter @count : i32",
        f"aiex.scratchpad_parameter @count : i32 {attributes}",
    )
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(text))
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[runtime("seq")])
    composed = Program.compose([built, prebuilt], entry=prebuilt.runtimes[0])
    assert f"aiex.scratchpad_parameter @count : i32 {attributes}" in str(
        composed.resolve_program()
    )


def test_a_lowered_prebuilt_declaration_and_a_built_one_of_one_name_conflict():
    text = PREBUILT.replace(
        "aiex.scratchpad_parameter @count : i32",
        "aiex.scratchpad_parameter @count : i32 {kind = 1 : i32, state_table_idx = 3 : ui8}",
    )
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(text))
    rt = runtime("seq")
    rt.register_parameter(ScratchpadParameter("count", np.int32))
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[rt])
    with pytest.raises(
        ValueError,
        match=r"ScratchpadParameter 'count' is declared with attributes \{\} and "
        r"\{'kind': '1 : i32', 'state_table_idx': '3 : ui8'\}",
    ):
        Program.compose([built, prebuilt], entry=rt).resolve_program()


def test_a_prebuilt_device_with_no_sequence_declares_what_its_cores_read():
    text = """
aiex.scratchpad_parameter @count : i32
aie.device(npu2_1col) @made_elsewhere {
  %t = aie.tile(0, 2)
  %c = aie.core(%t) {
    %v = aiex.read_scratchpad_parameter @count : i32
    aie.end
  }
}
"""
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(text))
    assert prebuilt.runtimes == []
    rt = runtime("seq")
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[rt])
    composed = Program.compose([built, prebuilt], entry=rt).resolve_program()
    assert composed.operation.verify()
    assert str(composed).count("aiex.scratchpad_parameter @count") == 1


def test_a_prebuilt_sequence_with_a_dynamic_shape_is_refused():
    text = PREBUILT.replace("%a: memref<16xi32>", "%a: memref<?xi32>")
    with Context(), pytest.raises(TypeError, match="has a dynamic shape"):
        DeviceConfiguration.from_module("other", Module.parse(text))


def test_a_prebuilt_device_may_read_its_scratchpad_parameters():
    text = PREBUILT.replace(
        "    aie.end",
        "    %v = aiex.read_scratchpad_parameter @count : i32\n    aie.end",
    )
    with Context():
        prebuilt = DeviceConfiguration.from_module("other", Module.parse(text))
    built = DeviceConfiguration("built", NPU2Col1(), runtimes=[runtime("seq")])
    (run,) = prebuilt.runtimes
    composed = Program.compose([built, prebuilt], entry=run).resolve_program()
    assert composed.operation.verify()
    text = str(composed)
    assert text.count("aiex.scratchpad_parameter @count") == 1
    assert "aiex.read_scratchpad_parameter @count" in text


@pytest.mark.parametrize(
    "text", ["", PREBUILT + PREBUILT.split("\n", 2)[2].replace("made_", "built_")]
)
def test_a_prebuilt_module_holds_one_device(text):
    with Context(), pytest.raises(ValueError, match="holding one aie.device, not"):
        DeviceConfiguration.from_module("other", Module.parse(text))


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
    assert merged.arg_types == [Line, Line]
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[worker], runtimes=[merged]
    )
    text = str(Program.compose([config], entry=merged).resolve_program())
    sequence = text[text.index("aie.runtime_sequence") :]
    steps = re.findall(r"aiex\.dma_(start|await|free)_task", sequence)
    assert steps == ["start", "await", "free", "start", "free"]


def chained_steps():
    """A step that only fills and one that drains what the core makes of it."""
    of_in, of_out = ObjectFifo(Line, name="into"), ObjectFifo(Line, name="out_of")
    worker = Worker(lambda a, b: None, [of_in.cons(), of_out.prod()])
    fill = Runtime(lambda c, h: h.fill(c), [Line, of_in.prod()])
    drain = Runtime(lambda c, h: h.drain(c, wait=True), [Line, of_out.cons()])
    return worker, fill, drain


def merged_transfers(worker, merged):
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[worker], runtimes=[merged]
    )
    text = str(Program.compose([config], entry=merged).resolve_program())
    sequence = text[text.index("aie.runtime_sequence") :]
    return re.findall(r"aiex\.dma_(start|await|free)_task", sequence)


def test_merged_steps_close_their_groups_in_turn():
    # The fill is freed before the drain that completes it is issued.
    worker, fill, drain = chained_steps()
    merged = Runtime.merge([fill, drain])
    assert merged_transfers(worker, merged) == [
        "start",
        "free",
        "start",
        "await",
        "free",
    ]


def test_merge_run_issues_chained_steps_in_one_group():
    worker, fill, drain = chained_steps()

    def together(calls):
        (_, (a, into)), (_, (b, out_of)) = calls
        tg = TaskGroup()
        into.fill(a, group=tg)
        out_of.drain(b, wait=True, group=tg)
        tg.finish()

    merged = Runtime.merge([fill, drain], run=together)
    assert merged.arg_types == [Line, Line]
    assert merged_transfers(worker, merged) == [
        "start",
        "start",
        "await",
        "free",
        "free",
    ]


def syncing_step(name):
    """A core reading its own parameter, and a step that syncs and drains it."""
    count = ScratchpadParameter(f"count_{name}", np.int32)
    of_out = ObjectFifo(Line, name=f"out_{name}")

    def core(out, count):
        elem = out.acquire(1)
        elem[0] = count.read()
        out.release(1)

    def sequence(c, out):
        sync_parameters()
        out.drain(c, wait=True)

    worker = Worker(core, [of_out.prod(), count], while_true=False)
    return worker, Runtime(sequence, [Line, of_out.cons()])


def test_merged_steps_sync_their_parameters_once():
    (worker_a, step_a), (worker_b, step_b) = syncing_step("a"), syncing_step("b")
    merged = Runtime.merge([step_a, step_b])
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[worker_a, worker_b], runtimes=[merged]
    )
    module = Program.compose([config], entry=merged).resolve_program()
    PassManager.parse(
        "builtin.module(aie-resource-stages,aie-lower-scratchpad-parameters)",
        module.context,
    ).run(module.operation)
    text = str(module)
    sequence = text[text.index("aie.runtime_sequence") :]
    assert sequence.count("aiex.npu.create_scratchpad") == 1
    # One release of each core's sync lock.
    assert sequence.count("aiex.set_lock") == 2


def test_sync_under_a_branch_is_refused():
    count = ScratchpadParameter("count_branch", np.int32)
    of_out = ObjectFifo(Line, name="out_branch")

    def core(out, count):
        elem = out.acquire(1)
        elem[0] = count.read()
        out.release(1)

    def sequence(c, out):
        # A sync the branch may skip must not stand in for the one after it.
        with if_(True):
            sync_parameters()
        sync_parameters()
        out.drain(c, wait=True)

    worker = Worker(core, [of_out.prod(), count], while_true=False)
    rt = Runtime(sequence, [Line, of_out.cons()])
    config = DeviceConfiguration("dev", NPU2Col1(), workers=[worker], runtimes=[rt])
    with pytest.raises(ValueError, match="top level of a runtime sequence body"):
        Program.compose([config], entry=rt).resolve_program()


def waiting_worker(name, hops=0):
    """Build a Worker that waits for its parameter each pass, and the fifo the host drains.

    Args:
        name: Suffix for the parameter and fifo names.
        hops: Memtile forwards between the core and the host's fifo.

    Returns:
        The Worker and the ObjectFifo the host drains.
    """
    count = ScratchpadParameter(f"count_{name}", np.int32)
    of_out = ObjectFifo(Line, name=f"out_{name}")

    def core(out, count):
        elem = out.acquire(1)
        elem[0] = count.read()
        out.release(1)

    worker = Worker(core, [of_out.prod(), count], wait_for_parameters=True)
    for hop in range(hops):
        of_out = of_out.cons().forward(name=f"out_{name}_{hop}")
    return worker, of_out


def syncing_drain(of_out):
    def sequence(c, out):
        sync_parameters()
        out.drain(c, wait=True)

    return Runtime(sequence, [Line, of_out.cons()])


@pytest.mark.parametrize("hops", [0, 2])
def test_merge_refuses_two_syncing_steps_on_one_waiting_worker(hops):
    worker, of_out = waiting_worker("w", hops)
    merged = Runtime.merge([syncing_drain(of_out), syncing_drain(of_out)])
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[worker], runtimes=[merged]
    )
    with pytest.raises(
        ValueError, match=r"steps 0 \(.*\) and 1 \(.*\) both sync parameters"
    ):
        Program.compose([config], entry=merged).resolve_program()


def test_merge_keeps_syncing_steps_on_different_waiting_workers():
    (worker_a, of_a), (worker_b, of_b) = waiting_worker("a"), waiting_worker("b")
    merged = Runtime.merge([syncing_drain(of_a), syncing_drain(of_b)])
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[worker_a, worker_b], runtimes=[merged]
    )
    text = str(Program.compose([config], entry=merged).resolve_program())
    assert text.count("aiex.sync_scratchpad_parameters_from_host") == 1


def step_runtimes(n, out_of_cons):
    """``n`` Runtimes, each filling ``into`` and draining ``out_of``."""
    of_in, of_out = ObjectFifo(Line, name="into"), ObjectFifo(Line, name="out_of")
    worker = Worker(lambda a, b: None, [of_in.cons(), of_out.prod()])

    def step(a, b, into, out_of):
        into.fill(a)
        out_of.drain(b, wait=True)

    steps = [
        Runtime(step, [Line, Line, of_in.prod(), out_of_cons(of_out, k)], name=f"s{k}")
        for k in range(n)
    ]
    return worker, steps


def out_of_consumers(worker, steps, device):
    config = DeviceConfiguration("pack", device, workers=[worker], runtimes=steps)
    text = str(Program.compose([config], entry=steps[0]).resolve_program())
    (consumers,) = re.findall(r"aie\.objectfifo @out_of\(%\w+, \{([^}]*)\}", text)
    return consumers.split(", ")


def test_runtimes_draining_one_fifo_share_its_shim_end():
    steps = step_runtimes(3, lambda of, k: of.cons())
    assert len(out_of_consumers(*steps, NPU2Col1())) == 1


def test_runtimes_draining_one_fifo_on_the_same_channel_share_its_shim_end():
    steps = step_runtimes(3, lambda of, k: of.cons(port=Port.dma(0)))
    assert len(out_of_consumers(*steps, NPU2Col1())) == 1


def test_runtimes_draining_one_fifo_on_unpinned_dma_ports_share_its_shim_end():
    # cons() and cons(port=Port.dma()) both leave the channel to allocation.
    steps = step_runtimes(2, lambda of, k: of.cons(port=Port.dma()) if k else of.cons())
    assert len(out_of_consumers(*steps, NPU2Col1())) == 1


def test_runtimes_draining_one_fifo_to_distinct_shim_tiles_keep_both():
    steps = step_runtimes(
        2, lambda of, k: of.cons(tile=Tile(k, 0, tile_type=AIETileType.ShimNOCTile))
    )
    assert len(out_of_consumers(*steps, NPU2())) == 2


def test_runtimes_draining_one_shim_end_at_different_depths_are_refused():
    with pytest.raises(ValueError, match="'out_of' is drained to one shim end"):
        step_runtimes(2, lambda of, k: of.cons(depth=k + 1))


def test_runtimes_draining_one_shim_end_at_different_channels_are_refused():
    with pytest.raises(ValueError, match="'out_of' is drained to one shim end"):
        step_runtimes(2, lambda of, k: of.cons(port=Port.dma(k)))


def test_merged_steps_may_each_use_only_explicit_groups():
    # Each step's default group is left empty, and must not stay open.
    of_in, of_out = ObjectFifo(Line, name="into"), ObjectFifo(Line, name="out_of")
    worker = Worker(lambda a, b: None, [of_in.cons(), of_out.prod()])

    def step(c, into, out_of):
        tg = TaskGroup()
        into.fill(c, group=tg)
        out_of.drain(c, wait=True, group=tg)
        tg.finish()

    steps = [Runtime(step, [Line, of_in.prod(), of_out.cons()]) for _ in range(3)]
    merged = Runtime.merge(steps)
    config = DeviceConfiguration(
        "pack", NPU2Col1(), workers=[worker], runtimes=[merged]
    )
    text = str(Program.compose([config], entry=merged).resolve_program())
    assert text.count("aiex.dma_start_task") == 6


def test_a_configured_runtime_does_not_merge():
    rt = runtime("seq")
    DeviceConfiguration("dev", NPU2Col1(), runtimes=[rt])
    with pytest.raises(ValueError, match="merge it before configuring"):
        Runtime.merge([rt, runtime("other")])
