# Logical Notes

## Devices

Device with ownership of specific #s or 

```python
dev = NPU1Col1()

# Note: using compute in examples, but could work for mem and shim too

# Still allow specific placement, when desired
specific_compute_tile = dev.checkout_compute_tile(col=0, row=1)

# Get a handle to a unique resource with knowing coordinate
compute_iter = dev.compute_iter()
my_compute_tile = compute_iter.next() # AnyTile interface

# Get a (unique) placeholder tile
some_compute_tile = dev.checkout_any_compute()
# I can use this in more than one place to signify co-location

# Don't enforce uniqueness - this would be default.
any_compute_tile = dev.any_compute()
```

# Current (placed) example code:
```python
line_size = vector_size // 4
line_type = np.ndarray[np.uint8, (line_size,)]

of_in = MyObjectFifo(2, line_type, shim_endpoint=(0, 0))
of_out = MyObjectFifo(2, line_type, shim_endpoint=(0, 0))

passthrough_fn = BinKernel(
    "passThroughLine",
    "passThrough.cc.o",
    [line_type, line_type, np.int32],
)


def core_fn(of_in, of_out, passThroughLine):
    for _ in range_(vector_size // line_size):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

worker_program = MyWorker(
    core_fn, [of_in.second, of_out.first, passthrough_fn], coords=(0, 2)
)
inout_sequence = SimpleFifoInOutSequence(
    of_in.first, vector_size, of_out.second, vector_size
)

my_program = MyProgram(
    NPU1Col1(), worker_programs=[worker_program], inout_sequence=inout_sequence
)
my_program.resolve_program()
```

# Hypothetical (more logical) Rewrite:
```python
line_size = vector_size // 4
line_type = np.ndarray[np.uint8, (line_size,)]

of_in = MyObjectFifo(2, line_type) # Can default to AnyShim for endpoint if endpoint isn't core
of_out = MyObjectFifo(2, line_type) # Can default to AnyShim for endpoint if endpoint isn't core

passthrough_fn = BinKernel(
    "passThroughLine",
    "passThrough.cc.o",
    [line_type, line_type, np.int32],
)


def core_fn(of_in, of_out, passThroughLine):
    for _ in range_(vector_size // line_size):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

worker_program = MyWorker(
    core_fn, [of_in.second, of_out.first, passthrough_fn] # Can omit placement, default to AnyCompute
)
inout_sequence = SimpleFifoInOutSequence(
    of_in.first, vector_size, of_out.second, vector_size
)

my_program = MyProgram(
    NPU1Col1(), # Device
    worker_programs=[worker_program], inout_sequence=inout_sequence,
    placer=SequentialPlacer # Placement class, initialized with device, probably with method like place_components(program_information)
)

# The placer is called during the resolve process, so the emitted ops are all placed
my_program.resolve_program()
```