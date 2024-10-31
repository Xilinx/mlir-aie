# Two workers with two functions
# fmt: off
of_in = ObjectFifo(2, tile_ty, "in0")
of_middle = ObjetctFifo(2, tile_ty, "middle")
of_out = ObjectFifo(2, tile_ty, "out0")


def core_add_fn(of_in1, of_out1):
    elem_in = of_in1.acquire(1)
    elem_out = of_out1.acquire(1)
    for i in range_(TILE_SIZE):
        elem_out[i] = elem_in[i] + 1
    of_in1.release(1)
    of_out1.release(1)


def core_double_fn(of_in1, of_out1):
    elem_in = of_in1.acquire(1)
    elem_out = of_out1.acquire(1)
    for i in range_(TILE_SIZE):
        elem_out[i] = elem_in[i] + 1
    of_in1.release(1)
    of_out1.release(1)


add_worker = Worker(
    core_add_fn, fn_args=[of_in.cons, of_middle.prod], while_true=True
)
double_worker = Worker(
    core_double_fn, fn_args=[of_middle.cons, of_out.prod], while_true=True
)


# Function composition through composing function + new worker
def core_fn(of_in1, of_out1, of_in2, of_out2):
    core_add_fn(of_in1, of_out1)
    core_double_fn(of_in2, of_out2)


composed_worker = Worker(
    core_fn,
    fn_args=[of_in.cons, of_middle.prod, of_middle.cons, of_out.prod],
    while_true=True,
)


# Sequential fuse idea?
composed_worker = add_worker.sequential_fuse(double_worker)
# fmt: on