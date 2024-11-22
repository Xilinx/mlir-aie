import matplotlib.animation as animation
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from .utils import ceildiv


def animate_from_accesses(
    access_order_tensors: list[np.ndarray],
    access_count_tensors: list[np.ndarray] | None,
    title: str = "Animated Access Visualization",
) -> animation.FuncAnimation:
    if len(access_order_tensors) < 1:
        raise ValueError("At least one access order tensor is required.")
    if not (access_count_tensors is None):
        if len(access_count_tensors) < 1:
            raise ValueError(
                "access_count_tensor should be None or requires at least one tensor"
            )
        if len(access_count_tensors) != len(access_order_tensors):
            raise ValueError(
                "Number of access count tensors and number of access order tensors should be equal"
            )

    tensor_height, tensor_width = access_order_tensors[0].shape
    fig_width = 7
    if tensor_width < 32:
        fig_width = 5
    height_width_ratio = ceildiv(tensor_height, tensor_width)
    fig_height = min(fig_width, fig_width * height_width_ratio)

    if not (access_count_tensors is None):
        fig_height *= 2
        fig, (ax_order, ax_count) = plt.subplots(2, 1)
    else:
        fig, ax_order = plt.subplots()

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.suptitle(title)
    xs = np.arange(access_order_tensors[0].shape[1])
    ys = np.arange(access_order_tensors[0].shape[0])

    ax_order.xaxis.tick_top()
    ax_order.invert_yaxis()
    ax_order.set_title("Access Order Animation")

    if not (access_count_tensors is None):
        ax_count.xaxis.tick_top()
        ax_count.invert_yaxis()
        ax_count.set_title(f"Access Counts")

    def animate_order(i):
        access_heatmap = ax_order.pcolormesh(access_order_tensors[i])

        if not (access_count_tensors is None):
            count_heatmap = ax_count.pcolormesh(
                xs, ys, access_count_tensors[i], cmap="gnuplot2"
            )
            return (
                access_heatmap,
                count_heatmap,
            )
        return access_heatmap

    _animation = animation.FuncAnimation(
        fig,
        animate_order,
        frames=len(access_order_tensors),
        interval=max(400, 100 + 5 * len(access_order_tensors)),
    )

    plt.tight_layout()
    plt.close()
    return _animation


def visualize_from_accesses(
    access_order_tensor: np.ndarray,
    access_count_tensor: np.ndarray | None,
    title: str = "Access Visualization",
    show_arrows: bool | None = None,
    file_path: str | None = None,
    show_plot: bool = True,
):
    tensor_height, tensor_width = access_order_tensor.shape
    if tensor_height * tensor_width >= 1024:
        if show_arrows:
            print(
                f"show_arrows not recommended for tensor sizes > 1024 elements",
                file=sys.stderr,
            )
        if show_arrows is None:
            show_arrows = False
    elif show_arrows is None:
        # Set to true by default only for 'small' tensor sizes
        show_arrows = True

    fig_width = 7
    if tensor_width < 32:
        fig_width = 5
    height_width_ratio = ceildiv(tensor_height, tensor_width)
    fig_height = min(fig_width, fig_width * height_width_ratio)

    if not (access_count_tensor is None):
        fig_height *= 2
        fig, (ax_order, ax_count) = plt.subplots(2, 1)
    else:
        fig, ax_order = plt.subplots()

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.suptitle(title)
    xs = np.arange(access_order_tensor.shape[1])
    ys = np.arange(access_order_tensor.shape[0])

    _access_heatmap = ax_order.pcolormesh(xs, ys, access_order_tensor, cmap="gnuplot2")
    ax_order.xaxis.tick_top()
    ax_order.invert_yaxis()
    ax_order.set_title("Access Order")

    if not (access_count_tensor is None):
        max_count = np.max(access_count_tensor)
        _count_heatmap = ax_count.pcolormesh(
            xs, ys, access_count_tensor, cmap="gnuplot2"
        )
        ax_count.xaxis.tick_top()
        ax_count.invert_yaxis()
        ax_count.set_title(f"Access Counts (max={max_count})")

    # Add arrows to show access order
    if show_arrows:
        # Thanks to https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
        # Thanks to tmdavison answer here https://stackoverflow.com/a/40890587/7871710

        order_dict = {}
        for i in range(access_order_tensor.shape[0]):
            for j in range(access_order_tensor.shape[1]):
                if access_order_tensor[i, j] != -1:
                    order_dict[access_order_tensor[i, j]] = (i, j)

        order_keys = list(order_dict.keys())
        order_keys.sort()
        for i in range(order_keys[0], order_keys[-1]):
            y1, x1 = order_dict[i]
            y2, x2 = order_dict[i + 1]
            ax_order.arrow(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                length_includes_head=True,
                head_width=0.1,
                head_length=0.15,
                overhang=0.2,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

    plt.tight_layout()
    if show_plot:
        plt.show()
    if file_path:
        if os.path.exists(file_path):
            print(
                f"Cannot save plot to {file_path}; file already exists",
                file=sys.stderr,
            )
        plt.savefig(file_path)
    plt.close()
