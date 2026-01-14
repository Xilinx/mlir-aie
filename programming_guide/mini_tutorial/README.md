# <ins>IRON Mini Tutorial</ins>

## <ins>Key Components: Workers, ObjectFifos, Runtime</ins>

IRON provides an unplaced (deferred placement) [API](../../python/iron/) for NPU programming. Below are examples describing AIE compute code and the Object FIFO data movement primitive:

Compute code using workers:
```python
# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

buffer = Buffer(
    data_ty,
    name="buff",
    initial_value=np.array(range(data_size), dtype=np.int32),
)

def core_fn(buff):
    for i in range_(data_size):
        buff[i] = buff[i] + 1

# Create a worker to perform the task
my_worker = Worker(core_fn, [buffer]) # placement can be enforced: placement=Tile(1, 3)
```
More on the Worker in [Section 1](../section-1/README.md) of the programming guide and in the [worker.py](../../python/iron/worker.py).

Object FIFO data movement primitive:
```python
# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(data_ty, name="in") # default depth is 2
```
More on the Object FIFO in [Section 2a](../section-2/section-2a/README.md) of the programming guide and in the [objectfifo.py](../../python/iron/dataflow/objectfifo.py).

The IRON code [example](./aie2p.py) in this mini tutorial details the different parts of an IRON design. More information on the Runtime in particular can be found in [Section 2d](../section-2/section-2d/README.md) of the programming guide.

## <u>Exercises</u>
1. Familiarize yourself with [exercise_1](./exercise_1/exercise_1.py). The code contains a single Worker which has an already instantiated local buffer that it sends out to external memory. Run `python3 exercise_1.py` to run the program and verify the output.

2. Modify the code in [exercise_1](./exercise_1/exercise_1.py) so that the Worker receives its input data from external memory instead of using the local buffer, i.e. a passthrough.

3. Modify the code in [exercise_1](./exercise_1/exercise_1.py) so that data is routed from external memory through a Mem tile and back only using ObjectFifos and without using Workers. For this, you will require the `forward()` function described in [Section 2b - Implicit Copy](../section-2/section-2b/03_Implicit_Copy/README.md) of the programming guide. Don't forget to `fill()` the input Object FIFO with data at runtime as described in [Section 2d](../section-2/section-2d/RuntimeTasks.md).

4. Modify the code in [exercise_1](./exercise_1/exercise_1.py) so that data is first routed from external memory through a Mem tile to a Worker, which performs the passthrough, then send the data back out.

5. Modify the code in [exercise_1](./exercise_1/exercise_1.py) such that the output of the Worker is routed through a Mem tile.

## <ins>Complex Data Movement Patterns: Broadcast, Split, Join</ins>

IRON designs can be scaled to use multiple Workers easily: 
```python
n_workers = 4

# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

def core_fn(...):
    # ...kernel code...

# Create workers to perform the tasks
workers = []
for _ in range(n_workers):
    workers.append(
        Worker(core_fn, [...])
    )

rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(*workers)
```
More on programming for multiple workers in [Section 2e](../section-2/section-2e/README.md) of the programming guide.

Complex data movement patterns such as broadcast, split or join are supported using the `ObjectFifo`, and in particular the `ObjectFifoHandle`s, which can be either producer or consumer handles. These are used to determine a broadcast pattern with multiple consumers.

Broadcast - further documentation on the broadcast available in [Section 2b - Broadcast](../section-2/section-2b/02_Broadcast/) of the programming guide.
```python
n_workers = 4

# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(data_ty, name="in")

def core_fn(of_in):
    # ...kernel code...

# Create workers to perform the tasks
workers = []
for _ in range(n_workers):
    workers.append(
        Worker(core_fn, [of_in.cons()]) # each call to of_in.cons() returns a new ObjectFifoHandle
    )
```

The `split()` and `join()` methods are used to create multiple output and input `ObjectFifos` respectively.
    
Split - further documentation on the `split()` available in [Section 2b - Implicit Copy](../section-2/section-2b/03_Implicit_Copy/) of the programming guide.
```python
n_workers = 4

# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
tile_size = data_size // n_workers
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_offsets = [tile_size * worker for worker in range(n_workers)]

of_in = ObjectFifo(data_ty, name="in")
of_ins = of_in.cons().split(
    of_offsets,
    obj_types=[tile_ty] * n_workers,
    names=[f"in{worker}" for worker in range(n_workers)],
)

def core_fn(of_in):
    # ...kernel code...

# Create workers to perform the tasks
workers = []
for worker in range(n_workers):
    workers.append(
        Worker(core_fn, [of_ins[worker].cons()])
    )
```

Join - further documentation on the `join()` available in [Section 2b - Implicit Copy](../section-2/section-2b/03_Implicit_Copy/) of the programming guide.
```python
n_workers = 4

# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
tile_size = data_size // n_workers
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_offsets = [tile_size * worker for worker in range(n_workers)]

of_out = ObjectFifo(data_ty, name="out")
of_outs = of_out.prod().join(
    of_offsets,
    obj_types=[tile_ty] * n_workers,
    names=[f"out{worker}" for worker in range(n_workers)],
)

def core_fn(of_out):
    # ...kernel code...

# Create workers to perform the tasks
workers = []
for worker in range(n_workers):
    workers.append(
        Worker(core_fn, [of_outs[worker].prod()])
    )
```
More on the Object FIFO data movement patterns in [Section 2b](../section-2/section-2b/README.md) of the programming guide.

## <u>Exercises</u>
1. Familiarize yourself with [exercise_2](./exercise_2/exercise_2.py). Modify the code in [exercise_2](./exercise_2/exercise_2.py) so that the input data is split between three workers and their outputs are joined before the final result is sent to external memory.

2. Modify the code in [exercise_2](./exercise_2/exercise_2.py) such that the data sizes for each worker are uneven, for example: tile_sizes = [8, 24, 16].

## <ins>Runtime Sequence</ins>

The arguments of the IRON runtime sequence describe buffers that will be available on the host side; the body of the sequence contains commands which describe how those buffers are moved into the AIE-array through `ObjectFifos`.

```python
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(data_ty, name="in")
of_out = ObjectFifo(data_ty, name="out")

rt = Runtime()
with rt.sequence(tile_ty, tile_ty) as (a_in, c_out):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=True)
```

Up to five buffers are supported in the runtime sequence, where the fifth is typically used for trace support. This is further described in [Section 4b](../section-4/section-4b/README.md) of the programming guide.

Runtime sequence commands are submitted to and executed by a dedicated command processor in order. The command processor will wait on commands that are set to `wait` until a token associated with their completion is generated. When all the commands in the runtime sequence have been executed the command processor sends an interrupt to the host processor.

IRON also supports grouping of runtime sequence commands using `task_group`s. Commands that are in the same group begin execution concurrently, and the completion of the group can be explicitly synchronized using the `finish_task_group()` command. These features can be combined to achieve an optimized grouping of waits for parallel tasks, as is shown in [this](../../programming_examples/basic/memcpy/README.md) programming example.

More on the runtime sequence in [Section 2d](../section-2/section-2d/RuntimeTasks.md) of the programming guide.

## <u>Exercises</u>

1. Familiarize yourself with [exercise_3](./exercise_3/exercise_3.py). Right now the design does a simple passthrough, i.e. `out_C = in_A`, and a token is issued by the the `drain()` command in the runtime sequence upon its completion. Switch the places of the `fill()` and `drain()` commands and run `python3 exercise_3.py`. Observe what happens.

2. Restore the code in [exercise_3](./exercise_3/exercise_3.py) to its original version. Modify the code in [exercise_3](./exercise_3/exercise_3.py) to do an addition of two input tensors from external memory, i.e `out_C = in_A + in_B`.

## <ins>Runtime Parameters and Barriers</ins>

IRON supports Runtime Parameters which are set and propagated to the Workers at runtime.
```python
n_workers = 4

# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Runtime parameters
rtps = []
for i in range(n_workers):
    rtps.append(
        Buffer(
            np.ndarray[(16,), np.dtype[np.int32]],
            name=f"rtp{i}",
            use_write_rtp=True,
        )
    )

def core_fn(rtp):
    runtime_parameter = rtp

# Create workers to perform the tasks
workers = []
for worker in range(n_workers):
    workers.append(
        Worker(core_fn, [rtps[worker]])
    )

rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):

    # Set runtime parameters
    def set_rtps(*args):
        for rtp in args:
            rtp[0] = 50
            rtp[1] = 255
            rtp[2] = 0

    rt.inline_ops(set_rtps, rtps)
```
To ensure that RTPs are not read prematurely, `WorkerRuntimeBarriers` can be used to synchronize a Worker with the runtime sequence:
```python
n_workers = 4

# Define tensor types
data_size = 256
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Runtime parameters
# ...

# Worker runtime barriers
workerBarriers = []
for _ in range(n_workers):
    workerBarriers.append(WorkerRuntimeBarrier())

def core_fn(rtp, barrier):
    barrier.wait_for_value(1)
    runtime_parameter = rtp

# Create workers to perform the tasks
workers = []
for worker in range(n_workers):
    workers.append(
        Worker(core_fn, [rtps[worker], workerBarriers[worker]])
    )

rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    # Set runtime parameters
    # ...
    rt.inline_ops(set_rtps, rtps)
    
    for worker in range(n_workers):
        rt.set_barrier(workerBarriers[worker], 1)
```
More on the runtime parameters and barriers in [Section 2d](../section-2/section-2d/RuntimeTasks.md) of the programming guide and in the [worker.py](../../python/iron/worker.py).

## <u>Exercises</u>
1. Familiarize yourself with [exercise_4](./exercise_4/exercise_4.py). Modify line 83 by setting: `USE_INPUT_VEC = False`. Run `python3 exercise_4.py`.

2. The design fails because the Worker reads the RTP before the runtime sets it. Modify the code in [exercise_4](./exercise_4/exercise_4.py) such that the Worker uses a `WorkerRuntimeBarrier` to wait for the RTP to be set.

## <ins>Advanced Topic: Data Layout Transformations</ins>

AIE array DMAs can perform on-the-fly data transformations. Transformations on each dimension are expressed as a (size, stride) pair. Dimensions are given from highest to lowest:
```python
[(size_2, stride_2), (size_1, stride_1), (size_0, stride_0)]
```

Data layout transformations can be viewed as a way to specify to the hardware which location in the data to access next and as such it is possible to model the access pattern using a series of nested loops. For example, the transformation using the strides and sizes from above can be expressed as:
```c
int *buffer;
for(int i = 0; i < size_2; i++)
    for(int j = 0; j < size_1; j++)
        for(int k = 0; k < size_0; k++)
            // access/store element at/to buffer[  i * stride_2
            //                                   + j * stride_1
            //                                   + k * stride_0]
```

To better support DMA on-the-fly data transformations **at runtime** IRON provides [taplib](../../python/helpers/taplib/) which provides the building blocks for `Tensor Access Pattern`s (`taps`). The sizes and strides are grouped together and the dimensions should be given from highest to lowest (up to 4 dimensions):
```python
tap = TensorAccessPattern(
    tensor_dims=(2, 3),
    offset=0,
    sizes=[size_1, size_0],
    strides=[stride_1, stride_0],
)
```
`taps` additionally have an offset and as such the `tensor_dims` may be smaller than the size of the original tensor.

A `TensorAccessPattern` can be applied to the `fill()` and `drain()` runtime operations:
```python
rt = Runtime()
with rt.sequence(data_ty, data_ty) as (a_in, c_out):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in, tap)
    rt.drain(of_out.cons(), c_out, tap, wait=True)
```

The `TensorAccessPattern` can be visualized in two ways:
- as a heatmap showing the order that elements are accessed
- as a heatmap showing the number of times each element in the tensor is accessed by the `TensorAccessPattern`

```python
tap.visualize(show_arrows=True, plot_access_count=True)
```

`taps` can be grouped in a `TensorAccessSequence` where each `tap` represents a different tile (from the tiling pattern):
```python
t0 = TensorAccessPattern((8, 8), offset=0, sizes=[1, 1, 4, 4], strides=[0, 0, 8, 1])
t1 = TensorAccessPattern((8, 8), offset=4, sizes=[1, 1, 4, 4], strides=[0, 0, 8, 1])
t2 = TensorAccessPattern((8, 8), offset=32, sizes=[1, 1, 4, 4], strides=[0, 0, 8, 1])

taps = TensorAccessSequence.from_taps([t0, t1, t2])
```
The `taps` can then be accessed in the sequence as an array:
```python
for t in taps:
```

Deducing the sizes and strides for the `tap` can be challenging for the user. `taplib` introduces the `TensorTiler2D` class to try and address this challenge. Tilers are an explorative feature which is designed to generate `taps` for common tiling patterns. The tiler returns the generated `taps` as a `TensorAccessSequence`:
```python
tensor_dims = (8, 8)
tile_dims = (4, 4)
simple_tiler = TensorTiler2D.simple_tiler(tensor_dims, tile_dims)
```
The simple tiler above takes a very straighforward approach to tiling and makes a vertical split of the data based on the given dimensions. More tilers are available in [tensortiler2d.py](../../python/helpers/taplib/tensortiler2d.py).

More on `taplib` in [tiling_exploration](../../programming_examples/basic/tiling_exploration/README.md).

`ObjectFifo`s can express DMA on-the-fly data transformations via their `dims_to_stream` and `dims_from_stream_per_cons` inputs. These inputs are structured as a list of pairs where each pair is expressed as (size, stride) for a dimension of the DMA transformation. The dimensions should be given from highest to lowest:
```python
dims = [(size_2, stride_2), (size_1, stride_1), (size_0, stride_0)]
of_out = ObjectFifo(data_ty, name="out", dims_to_stream=dims)
```
Offsets are currently not represented at the Object FIFO level and as such the dimensions should be applicable over the full size of the objects.

More on the Object FIFO data layout transformations in [Section 2c](../section-2/section-2c/README.md) of the programming guide.

## <u>Exercises</u>
1. Familiarize yourself with [exercise_5a](./exercise_5/exercise_5a/exercise_5a.py). Use a `tap` such that the data transformation performed on the input data at runtime matches the one shown in [ref_plot.png](./exercise_5/exercise_5a/ref_plot.png). Don't forget to add the `tap` to the runtime `fill()` operation. Before running the example modify line 83 to `USE_REF_VEC = False`. Run `python3 exercise_5a.py` to verify your answer.

2. Replace the `tap` you added in [exercise_5a](./exercise_5/exercise_5a/exercise_5a.py) with one generated by a `TensorTiler2D`. For this, you will require the `simple_tiler()` constructor defined in [tensortiler2d.py](../../python/helpers/taplib/tensortiler2d.py). Run `python3 exercise_5a.py` to verify your answer. You can also observe the two generated plots.

3. Modify the code in [exercise_5a](./exercise_5/exercise_5a/exercise_5a.py) such that the data transformations are applied directly on `of_out`, instead of at runtime. Run `python3 exercise_5a.py` to verify your answer.

4. Familiarize yourself with [exercise_5b](./exercise_5/exercise_5b/exercise_5b.py). Observe how the `taps` in the `TensorAccessSequence` differ slightly from the one in [exercise_5a](./exercise_5/exercise_5a/exercise_5a.py). Run `python3 exercise_5b.py` and observe the two generated plots.
