# <ins>Hands-on Tutorial for IRON 0.9</ins>

IRON 0.9 introduces an unplaced layer to the IRON API. As an example, below are the placed and unplaced versions of the AIE Object FIFOs and compute code:

Placed Object FIFO: 
```python
line_size = 256
line_type = np.ndarray[(line_size,), np.dtype[np.int32]]

A = tile(1, 3)
B = tile(2, 4)
of_in = object_fifo("in", A, B, 2, line_type)
```
Unplaced Object FIFO:
```python
# Define tensor types
line_size = 256
line_type = np.ndarray[(line_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(line_type, name="in") # default_depth is 2
```
More on the Object FIFO in [Section 2a](../section-2/section-2a/README.md) of the programming guide.

Placed compute code: 
```python
ComputeTile1 = tile(1, 3)
buff_in = buffer(ComputeTile1, data_ty, name="buff")

@core(ComputeTile1)
def core_body():
    for i in range_(data_size):
        buff_in[i] = buff_in[i] + 1
```
Unplaced compute code:
```python
buff = GlobalBuffer(data_ty, name="buff")

def core_fn(buff_in):
    for i in range_(data_size):
        buff_in[i] = buff_in[i] + 1

# Create a worker to perform the task
my_worker = Worker(core_fn, [buff]) # enforced placement: placement=Tile(1, 3)
```

We will now look at a code [example](./aie2.py) of IRON 0.9 and observe the different parts of the design.

## <u>Exercises</u>
1. Familiarize yourself with [exercise_1](./exercise_1/aie2.py). The code contains a single Worker which has an already instantiated local buffer that it sends out to external memory. Run `make run` to run the program and verify the output.

2. Run `make clean`. Modify the code in [exercise_1](./exercise_1/aie2.py) so that the Worker receives its input data from external memory instead of using the local buffer, i.e. a passthrough.

3. Run `make clean`. Modify the code in [exercise_1](./exercise_1/aie2.py) so that data is routed from external memory through a Mem tile and back only using ObjectFifos and without using Workers. For this, you will require the `forward()` function defined in [objectfifo.py](../../python/iron/dataflow/objectfifo.py).

IRON 0.9 designs can be scaled to use multiple Workers easily: 
```python
n_workers = 4

def core_fn(...):
    # ...kernel code...

# Create workers to perform the tasks
workers = []
for _ in range(n_workers):
    workers.append(
        Worker(core_fn, [...])
    )
```
Complex data movement patterns such as broadcast, split or join are supported using the `ObjectFifo`. In particular the `ObjectFifoHandles`, which can be either producer or consumer handles, are used to determine a broadcast pattern with multiple consumers:
```python
n_workers = 4

# Define tensor types
line_size = 256
line_type = np.ndarray[(line_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(line_type, name="in")

def core_fn(of_in):
    # ...kernel code...

# Create workers to perform the tasks
workers = []
for _ in range(n_workers):
    workers.append(
        Worker(core_fn, [of_in.cons()])
    )
```

The `split` and `join` methods are used to create multiple output and input `ObjectFifos` respectively.
    
Split:
```python
n_workers = 4

# Define tensor types
line_size = 256
line_type = np.ndarray[(line_size,), np.dtype[np.int32]]
tile_size = line_size // n_workers

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

Join:
```python
n_workers = 4

# Define tensor types
line_size = 256
line_type = np.ndarray[(line_size,), np.dtype[np.int32]]
tile_size = line_size // n_workers

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

## <u>Exercises</u>
4. Familiarize yourself with [exercise_2](./exercise_2/aie2.py). Modify the code in [exercise_2](./exercise_2/aie2.py) so that the input data is split between three workers and their outputs are joined before the final result is sent to external memory.

IRON 0.9 also supports the use of Runtime Parameters which are set and propagated to the Workers at runtime.
```python
n_workers = 4

# Runtime parameters
rtps = []
for i in range(n_workers):
    rtps.append(
        GlobalBuffer(
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
To ensure that RTPs are not read prematurely, IRON 0.9 introduces `WorkerRuntimeBarriers` which can be used to synchronize a Worker with the runtime sequence:
```python
n_workers = 4

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

## <u>Exercises</u>
5. Familiarize yourself with [exercise_3](./exercise_3/aie2.py). Run `make run`. 

6. The design fails because the Worker reads the RTP before the runtime sets it. Modify the code in [exercise_3](./exercise_3/aie2.py) such that the Worker uses a `WorkerRuntimeBarrier` to wait for the RTP to be set.
