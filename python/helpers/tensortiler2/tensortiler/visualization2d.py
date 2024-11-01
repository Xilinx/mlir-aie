import numpy as np
import os
import sys

from .utils import ceildiv


def visualize_from_access_tensors(
    access_order_tensor: np.ndarray,
    access_count_tensor: np.ndarray | None,
    title: str = "Access Order and Access Count",
    show_arrows: bool = True,
    show_numbers: bool = False,
    file_path: str | None = None,
    show_plot: bool = True,
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except:
        raise ImportError(
            "You must pip install matplotlib in order to render access graphs"
        )

    tensor_height, tensor_width = access_order_tensor.shape
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
    _access_heatmap = ax_order.pcolor(access_order_tensor, cmap="gnuplot2")

    # Thanks to https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
    # put the major ticks at the middle of each cell, (0, 0) in upper left corner
    ax_order.set_xticks(np.arange(access_order_tensor.shape[1]) + 0.5, minor=False)
    ax_order.set_yticks(np.arange(access_order_tensor.shape[0]) + 0.5, minor=False)
    ax_order.invert_yaxis()
    ax_order.xaxis.tick_top()
    ax_order.set_xticklabels(
        np.arange(0, access_order_tensor.shape[1]), minor=False, rotation="vertical"
    )
    ax_order.set_yticklabels(np.arange(0, access_order_tensor.shape[0]), minor=False)

    # Add arrows to show access order
    if show_arrows:
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
                x1 + 0.5,
                y1 + 0.5,
                x2 - x1,
                y2 - y1,
                length_includes_head=True,
                head_width=0.1,
                head_length=0.15,
                overhang=0.2,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
        ax_order.set_title("Access Order")

    if not (access_count_tensor is None):
        max_count = np.max(access_count_tensor)

        _count_heatmap = ax_count.pcolor(access_count_tensor, cmap="gnuplot2")
        # Thanks to https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
        # put the major ticks at the middle of each cell, (0, 0) in upper left corner
        ax_count.set_xticks(np.arange(access_count_tensor.shape[1]) + 0.5, minor=False)
        ax_count.set_yticks(np.arange(access_count_tensor.shape[0]) + 0.5, minor=False)
        ax_count.invert_yaxis()
        ax_count.xaxis.tick_top()
        ax_count.set_xticklabels(
            np.arange(0, access_count_tensor.shape[1]),
            minor=False,
            rotation="vertical",
        )
        ax_count.set_yticklabels(
            np.arange(0, access_count_tensor.shape[0]), minor=False
        )
        ax_count.set_title("Access Counts")

    # Add numbers to the plot
    if show_numbers:
        # Thanks to https://stackoverflow.com/questions/37719304/python-imshow-set-certain-value-to-defined-color
        # Thanks to tmdavison answer here https://stackoverflow.com/a/40890587/7871710
        for i in range(access_order_tensor.shape[0]):
            for j in range(access_order_tensor.shape[1]):
                c = access_order_tensor[i, j]
                if c != -1:
                    ax_order.text(
                        j + 0.45,
                        i + 0.45,
                        str(c),
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    )
                if not (access_count_tensor is None):
                    c = access_count_tensor[i, j]
                    ax_count.text(
                        j + 0.45,
                        i + 0.45,
                        str(c),
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    )

    # plt.title(title)
    if show_plot:
        plt.show()
    if file_path:
        if os.path.exists(file_path):
            print(
                f"Cannot save plot to {file_path}; file already exists",
                file=sys.stderr,
            )
        plt.savefig(file_path)
