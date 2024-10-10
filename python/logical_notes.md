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

of_in = MyObjectFifo(2, line_type)
of_out = MyObjectFifo(2, line_type)

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

inout_shim = MyShim(of_in.first, of_out.second, coords=(0, 2))
inout_sequence = SimpleFifoInOutSequence(
    of_in.first, vector_size, of_out.second, vector_size
)

my_program = MyProgram(
    NPU1Col1(),
    worker_programs=[worker_program],
    inout_sequence=inout_sequence,
    shims=inout_shim
)
my_program.resolve_program()
```

# Hypothetical (more logical) Rewrite:
```python
line_size = vector_size // 4
line_type = np.ndarray[np.uint8, (line_size,)]

of_in = MyObjectFifo(2, line_type)
of_out = MyObjectFifo(2, line_type)

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
inout_shim = MyShim(of_in.first, of_out.second) # Can omit placement, default to AnyShim
inout_sequence = SimpleFifoInOutSequence(
    of_in.first, vector_size, of_out.second, vector_size
)

my_program = MyProgram(
    NPU1Col1(), # Device
    worker_programs=[worker_program],
    inout_sequence=inout_sequence,
    shims=inout_shim,
    placer=SequentialPlacer # Placement class, initialized with device, probably with method like place_components(program_information)
)

# The placer is called during the resolve process, so the emitted ops are all placed
my_program.resolve_program()
```

# IO Config Ideas

```python
in_config = MyInputDataConfig(
    name = "in",

    # Input Options
    input_stream_form = vector_type,        # [Required] datatype
    ingress_transform = None ,              # [Optional] Shim tile DMA characteristics TODO: None | sizes/offset/strides | TilerHelper(tile_sizes, offsets, tile_skips...)
    
    # Output Options
    output_stream_form = subvector_type,    # [Required] input form must tile evenly into output form
    egress_transform = None,                # [Optional] Compute tile DMA characteristics

    # Data layover
    intermediate_form = subvector_type,     # [Optional] input must tile into intermediate must tile into output      
    intermediate_ingress_transform = None,  # [Optional] -> only if intermediate, Mem tile DMA characteristics
    intermediate_egress_transform = None,   # [Optional] -> only if intermediate, Mem tile DMA characteritics
    #intermediate_padding                   # [Optional] -> only if intermediate, Mem tile DMA capabilities
    #intermediate_split = None              # [Optional] -> for Distribute() semantics, can be int (num split, assume even) or offset list
    #intermediate_repeat = None             # [Optional] -> count

    # Shim configs?
    distribute = None,                     # [Optional] None (compiler choice) | NoDistribute | SequentialDistribute | ProportionalDistribute
)
# Note: could be written in JSON or yaml or similar or pickled for library use
# Alternate interface could be builder for subset of cases:
# in_config = ConfigBuilder("in", inputdtype = np.ndarray[...]) \
#   .tile(size=, offset=,..)
#   .ping_pong() # true or false
#   .l2(subtile_size=[])
#   .distributeToEndpoints(count or offset list)

out_config = MyOutputDataConfig(
    name = "out",

    # Input Options
    input_stream_form = vector_type,        # [Required] datatype
    ingress_transform = None,               # [Optional] Shim tile DMA characteristics
    
    # Output Options
    output_stream_form = subvector_type,    # [Required] input form must tile evenly into output form
    egress_transform = None,                # [Optional] Compute tile DMA characteristics

    # Data layover
    intermediate_form = subvector_type,     # [Optional] input must tile into intermediate must tile into output      
    #intermediate_ingress_transform = ...   # [Optional] -> only if intermediate, Mem tile DMA characteristics
    #intermediate_egress_transoform = ...   # [Optional] -> only if intermediate, Mem tile DMA characteritics
    #intermediate_padding = (offset, num)   # [Optional] -> only if intermediate, Mem tile DMA capabilities
    #intermediate_join = None               # [Optional] -> for Join() semantics, can be int (num split, assume even) or offset list

    # Shim configs?
    distribute = None,                     # [Optional] None (compiler choice) | NoDistribute | SequentialDistribute | ProportionalDistribute
)
io_config = InterleaveConfig(in_config, out_config) # sync_behvaior=??, varargs for # of configs to add
```

# IO Config Example

```python
import numpy as np
from aie.dialects.api.io import MyInputDataConfig, MyOutputDataConfig, MySequentialConfig
from aie.dialects.api.kernels import MyBinKernel
from aie.dialects.api import MyAIEJob
from aie.dialects.api.logical import MyDefaultPlacer

# Define types
vector_size = 4096
vector_type = np.ndarray[(vector_size,), np.dtype[np.uint8]]
subvector_size = vector_size // 4
subvector_type = np.ndarray[(subvector_size,), np.dtype[np.uint8]]

# Build information on input/output
in_config = MyInputDataConfig("in", in_type=vector_type, out_type=subvector_type)
out_config = MyOutputDataConfig("out", in_type=subvector_type, out_type=vector_type) # in_config.generate_reverse_config("out")?
io_config = MySequentialConfig(in_config, out_config) #, sync_behavior=EndOnly)

# Extract in/out fifos
endpoints = io_config.finalize()            # generate fifo endpoints & validate
of_in = endpoints["in"]                     # access fifo endpoints with dictionary
of_out = endpoints["out"]                   # access fifo endpoints with dictionary

passthrough_fn = MyBinKernel("passThroughLine", "passThrough.cc.o", [subvector_type, subvector_type, np.int32])

def core_fn(of_in, of_out, passThroughLine):
    for _ in range_(vector_size // line_size):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

MyAIEJob(NPU1Col1(), io_config) \
    .addWorker(core_fn, [of_in, of_out, passthrough_fn]) \
    # .addPlacer(MyDefaultPlacer())
    .emit()
