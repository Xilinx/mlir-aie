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
from .jit import compile_kernel_objects


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

    def compile(self):
        """
        Compile the graph operations into kernel objects.
        This prepares the graph for execution but doesn't run it yet.
        """
        # Create a mapping from captured tensors to their computed results
        tensor_map = {}
        kernel_objects = []  # Collect all kernel objects

        # Execute all operations and collect kernel objects
        kernel_name_counts = {}  # Track counts for each kernel name
        
        for i, node in enumerate(self.nodes):
            # Map inputs to their computed results
            mapped_inputs = []
            for inp in node.inputs:
                if inp in tensor_map:
                    mapped_inputs.append(tensor_map[inp])
                else:
                    mapped_inputs.append(inp)

            # Execute the operation - this returns a kernel object
            # Pass the output tensor as the last argument to the function
            kernel_object = node.func(*mapped_inputs, out=node.output)
            
            # Make kernel name unique by appending index
            original_name = kernel_object.kernel_name
            if original_name in kernel_name_counts:
                kernel_name_counts[original_name] += 1
                kernel_object.kernel_name = f"{original_name}_{kernel_name_counts[original_name]}"
            else:
                kernel_name_counts[original_name] = 0
                kernel_object.kernel_name = f"{original_name}_{kernel_name_counts[original_name]}"

            # Store the kernel object
            kernel_objects.append(kernel_object)
            
            # Store the output tensor for mapping
            tensor_map[node.output] = node.output

        print(f"Compiling {len(kernel_objects)} kernel objects")
        print(f"Kernel objects names: {[kernel_object.kernel_name for kernel_object in kernel_objects] if kernel_objects else []}")
        # Compile all kernel objects into a single UberKernel
        if kernel_objects:
            import pyxrt as xrt
            
            self.uber_kernel = compile_kernel_objects(kernel_objects)
            self.kernel_objects = kernel_objects
            
            # Create runlist for batch execution
            self.runlist = xrt.runlist(self.uber_kernel.get_hw_context())

            # Create runs for all kernel objects
            for kernel_object in kernel_objects:
                # Create run object using kernel object's stored parameters
                name = kernel_object.kernel_name
                kernel = self.uber_kernel.get_kernel(name)
                run = xrt.run(kernel["kernel"])
                run.set_arg(0, kernel_object.opcode)
                run.set_arg(1, self.uber_kernel.get_insts(name)["buffer_bo"])
                run.set_arg(2, self.uber_kernel.get_insts(name)["n_insts"])

                # Set additional kernel arguments from kernel object
                for i, arg in enumerate(kernel_object.kernel_args):
                    run.set_arg(3 + i, arg)

                # Add run to runlist
                self.runlist.add(run)
        else:
            self.uber_kernel = None
            self.kernel_objects = []
            self.runlist = None
        
        # Store tensor map for replay
        self.tensor_map = tensor_map

    def replay(self) -> Tensor:
        """
        Execute the compiled graph operations.
        Must call compile() first.

        Returns:
            Output tensor from the graph execution
        """
        if not hasattr(self, 'runlist') or self.runlist is None:
            raise RuntimeError("Graph must be compiled before replay. Call compile() first.")

        # Execute and wait
        self.runlist.execute()
        self.runlist.wait()
        
        # Return the output tensor from the last node
        return self.tensor_map[self.nodes[-1].output]


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
