# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
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
