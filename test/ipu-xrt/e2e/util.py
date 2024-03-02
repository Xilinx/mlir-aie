import __main__
import collections
import inspect
from itertools import islice, zip_longest
import os
from pathlib import Path

from aie.dialects import aie
from aie.extras.util import find_ops
from aie.ir import Context, InsertionPoint, Location, Module, UnitAttr

WORKDIR = os.getenv("WORKDIR")
if WORKDIR is None:
    WORKDIR = Path(__main__.__file__).parent.absolute() / (
        __main__.__file__[:-3] + "_workdir"
    )
else:
    WORKDIR = Path(WORKDIR).absolute()

WORKDIR.mkdir(exist_ok=True)


def construct_and_print_module(f):
    global WORKDIR
    assert WORKDIR is not None and WORKDIR.exists()
    WORKDIR = WORKDIR / (f.__name__ + "_workdir")
    WORKDIR.mkdir(exist_ok=True)

    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            args = inspect.getfullargspec(f).args
            if args:
                if args[0] in {"module", "_module"}:
                    module = f(module)
                else:
                    raise Exception(f"only `module` arg supported {args=}")
            else:
                f()
        if module is not None:
            assert module.operation.verify()
            print(module)


def grouper(iterable, n, *, incomplete="fill", fill_value=None):
    args = [iter(iterable)] * n
    match incomplete:
        case "fill":
            return zip_longest(*args, fillvalue=fill_value)
        case "strict":
            return zip(*args, strict=True)
        case "ignore":
            return zip(*args)
        case _:
            raise ValueError("Expected fill, strict, or ignore")


def sliding_window(iterable, n):
    it = iter(iterable)
    window = collections.deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)


def display_flows(module):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots()
    for c in find_ops(
        module.operation,
        lambda o: isinstance(o.operation.opview, aie.FlowOp),
    ):
        arrow = mpatches.FancyArrowPatch(
            (c.source.owner.opview.col.value, c.source.owner.opview.row.value),
            (c.dest.owner.opview.col.value, c.dest.owner.opview.row.value),
            mutation_scale=10,
        )
        axs.add_patch(arrow)

    axs.set(xlim=(-1, 5), ylim=(-1, 6))
    fig.show()
    fig.tight_layout()
    fig.savefig("flows.png")


def annot(op, annot):
    op.operation.attributes[annot] = UnitAttr.get()
