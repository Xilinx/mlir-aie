# graph.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

from contextlib import contextmanager
from typing import Generator, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from .tensor import Tensor


# Global graph capture state
_graph_capture_enabled = False
_captured_graph: List["GraphNode"] = []


@dataclass
class GraphNode:
    """Represents a single operation in the captured graph."""

    operation: str  # e.g., "matmul", "add", etc.
    func: Callable  # The actual function to execute
    inputs: List[Tensor]  # Input tensors
    output: Tensor  # Output tensor
    promise: Optional[Any] = None  # Promise object for kernel operations
    metadata: Dict[str, Any] = None  # Additional operation metadata


class Graph:
    """Represents a captured computation graph."""

    def __init__(self, nodes: List[GraphNode]):
        self.nodes = nodes

    def replay(self) -> List[Tensor]:
        """
        Replay the graph operations using the current data in input tensors.
        Uses runlist to batch all kernel executions for better performance.

        Returns:
            List of output tensors from the graph execution
        """
        import pyxrt as xrt

        # Create a mapping from captured tensors to their computed results
        tensor_map = {}
        promises = []  # Collect all Promise objects

        # Execute all operations and collect Promise objects
        for node in self.nodes:
            # Map inputs to their computed results
            mapped_inputs = []
            for inp in node.inputs:
                if inp in tensor_map:
                    mapped_inputs.append(tensor_map[inp])
                else:
                    mapped_inputs.append(inp)

            # Execute the operation - this returns a Promise object
            promise = node.func(*mapped_inputs)

            # Store the Promise object
            promises.append(promise)

            # Store the output tensor for mapping
            tensor_map[node.output] = node.output

        # Execute all Promise objects using runlist if there are any
        if promises:
            # Create a runlist for batch execution
            runlist = xrt.runlist()

            # Create runs for all Promise objects
            for promise in promises:
                # Create run object using Promise's stored parameters
                run = xrt.run(promise.kernel)
                run.set_arg(0, promise.opcode)
                run.set_arg(1, promise.insts_buffer_bo)
                run.set_arg(2, promise.n_insts)

                # Set additional kernel arguments from Promise
                for i, arg in enumerate(promise.kernel_args):
                    run.set_arg(3 + i, arg)

                # Add run to runlist
                runlist.add(run)

            # Execute all kernels in the runlist
            runlist.execute()

            # Wait for completion and process results
            runlist.wait()

            # Check state
            state = runlist.state()
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"Kernel batch returned {state}")

            # Mark all Promise objects as completed
            for promise in promises:
                promise.kernel_handle = runlist  # Store reference for done() method

        # Collect all outputs in order
        outputs = []
        for node in self.nodes:
            if node.output in tensor_map:
                outputs.append(tensor_map[node.output])
            else:
                # This shouldn't happen, but handle gracefully
                outputs.append(node.output)

        return outputs

    def __len__(self) -> int:
        """Return the number of operations in the graph."""
        return len(self.nodes)

    def __iter__(self):
        """Iterate over the graph nodes."""
        return iter(self.nodes)


def is_graph_capture_enabled() -> bool:
    """Check if graph capture mode is currently enabled."""
    return _graph_capture_enabled


def get_captured_graph() -> List[GraphNode]:
    """Get the currently captured graph."""
    return _captured_graph.copy()


def clear_captured_graph() -> None:
    """Clear the captured graph."""
    global _captured_graph
    _captured_graph.clear()


def add_to_graph(
    operation: str, func: Callable, inputs: List[Tensor], output: Tensor, promise: Optional[Any] = None, **metadata
) -> None:
    """Add an operation to the captured graph."""
    if _graph_capture_enabled:
        node = GraphNode(
            operation=operation,
            func=func,
            inputs=inputs,
            output=output,
            promise=promise,
            metadata=metadata,
        )
        _captured_graph.append(node)




def execute_graph() -> List[Tensor]:
    """
    Execute the captured graph by running each operation's function.

    Returns:
        List of output tensors from the graph execution
    """
    outputs = []
    for node in _captured_graph:
        # Execute the function with the captured inputs
        result = node.func(*node.inputs)
        outputs.append(result)
    return outputs


@contextmanager
def capture_graph() -> Generator[Graph, None, None]:
    """
    Context manager for graph capture mode.

    When used as a context manager, enables graph capture mode for the duration
    of the context and returns a Graph object that can be executed.

    Example:
        with capture_graph() as graph:
            # Operations here will be captured in graph mode
            result = matmul(a, b)

        # Execute the captured graph
        outputs = graph.replay()
    """
    global _graph_capture_enabled, _captured_graph

    # Store previous state
    previous_state = _graph_capture_enabled

    try:
        # Enable graph capture mode and clear existing graph
        _graph_capture_enabled = True
        _captured_graph.clear()

        # Create and yield the graph object
        graph = Graph(_captured_graph)
        yield graph
    finally:
        # Restore previous state
        _graph_capture_enabled = previous_state
