import collections
from itertools import islice, zip_longest

from aie.dialects import aie
from aie.extras.util import find_ops
from aie.ir import UnitAttr


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
