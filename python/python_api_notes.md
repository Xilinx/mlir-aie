# Python API Brainstorming

Passthrough Kernel Example Idea:

```python
vector_size = ...
line_size = vector_size // 4
line_type = np.ndarray[(line_size,), np.dtype[np.uint8]]

io = IOPolicy()
of_in = IOFifo(2, line_type, io)
of_out = IOFifo(2, line_type, io)

tiler = DataTiler((vector_size,))
for t in io.tile_loop(tiler):
    of_in.fill(t)
    of_out.drain(t)

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

my_program = MyProgram(NPU1Col1(), io, [core_fn, [of_in, of_out, passthrough_fn]])
my_program.validate()                 # Validates construction
print(my_program.resolve_program())   # Outputs mlir

A: np.ndarray[(vector_size,), np.dtype[np.uint8]] = ...
C: np.ndarray[(vector_size,), np.dtype[np.uint8]] = ...
my_program.run(inputs=A, outputs=C)
```

# More complex brainstorming?

```python
io = IOPolicy(placement_policy=Interleave | Split | Single, task_group_policy=Tiler  | Subtiler ?)
of_in = IOFifo(…, io)
of_out = IOFifo(…, io)

T = DataTiler((H, W), (h, w)) # TODO: likely will have several ways to express data
For t in io.TileLoop(T):
	S = t.subtiler((h1, w1), (h1/2, w1))
	For s in io.SubTileLoop(S, placement=?):
		of_in.fill(s) # repeatCount = n??
		of_out.drain(s)
of_in_handles = of_in.split(offsets)
of_out_handles = of_out.join(offsets) 
OR
of_in_handle = of_in.handle()
```
