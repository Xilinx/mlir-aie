import torch
import torch.nn.functional as F
from contextlib import contextmanager
import logging
import numpy as np


from aie.iron import transform

# Global state
graph_operations = []
is_capturing = False
tensor_refs = {}  # Store references to intermediate tensors
next_tensor_id = 0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def to_torch_tensor(tensor):
    """
    Helper function to convert AIE Iron tensors to PyTorch tensors.

    Args:
        tensor: AIE Iron tensor or any tensor-like object with .numpy() method

    Returns:
        torch.Tensor: PyTorch tensor
    """
    if isinstance(tensor, str):
        # During graph capturing, tensor references are strings
        # We should return them as-is for capturing, but this shouldn't be called during execution
        if is_capturing:
            return tensor
        else:
            # During execution, try to get the actual tensor from tensor_refs
            if tensor in tensor_refs and tensor_refs[tensor] is not None:
                return tensor_refs[tensor]
            else:
                raise ValueError(f"Tensor reference {tensor} not found in tensor_refs")
    elif hasattr(tensor, "numpy"):
        # Convert AIE Iron tensor to numpy first, then to torch
        numpy_array = tensor.numpy()
        return torch.from_numpy(numpy_array)
    elif isinstance(tensor, torch.Tensor):
        # Already a torch tensor, return as is
        return tensor
    elif isinstance(tensor, np.ndarray):
        # Already a numpy array, convert to torch
        return torch.from_numpy(tensor)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


def format_tensor_values(tensor, name, max_values=10):
    """Helper function to format tensor values for logging"""
    values = [round(x, 3) for x in tensor.flatten()[:max_values].tolist()]
    return f"     {name}: {tensor.shape}, first {max_values} values: {values}"


def matmul(a, b, device="cpu"):
    """
    Matrix multiplication between tensors a and b
    """

    # Convert inputs to PyTorch tensors
    a = to_torch_tensor(a)
    b = to_torch_tensor(b)

    if is_capturing:
        global next_tensor_id
        # Create symbolic tensor reference
        tensor_id = f"tensor_{next_tensor_id}"
        next_tensor_id += 1
        tensor_refs[tensor_id] = None  # Will be computed during execution

        # Record the operation with tensor references
        graph_operations.append((matmul, a, b, device, tensor_id))
        logger.debug(f"captured matmul -> {tensor_id}")

        # Return symbolic tensor reference
        return tensor_id

    result = torch.matmul(a, b)

    return result


def silu(x, device="cpu"):
    """
    SiLU (Sigmoid Linear Unit) activation function
    This is the key activation used in Llama2
    """

    # Convert input to PyTorch tensor
    x = to_torch_tensor(x)

    if is_capturing:
        global next_tensor_id
        # Create symbolic tensor reference
        tensor_id = f"tensor_{next_tensor_id}"
        next_tensor_id += 1
        tensor_refs[tensor_id] = None  # Will be computed during execution

        # Record the operation with tensor references
        graph_operations.append((silu, x, device, tensor_id))
        logger.debug(f"captured silu -> {tensor_id}")

        # Return symbolic tensor reference
        return tensor_id

    result = F.silu(x)

    return result


def transform(a, b, operation, device="cpu"):
    """
    Apply element-wise operation between tensors a and b
    """

    # Convert inputs to PyTorch tensors
    a = to_torch_tensor(a)
    b = to_torch_tensor(b)

    if is_capturing:
        global next_tensor_id
        # Create symbolic tensor reference
        tensor_id = f"tensor_{next_tensor_id}"
        next_tensor_id += 1
        tensor_refs[tensor_id] = None  # Will be computed during execution

        # Record the operation with tensor references
        graph_operations.append((transform, a, b, operation, device, tensor_id))
        logger.debug(f"captured transform -> {tensor_id}")

        # Return symbolic tensor reference
        return tensor_id

    result = operation(a, b)
    return result


def set_verbose(level):
    """
    Set the verbose level for logging operations
    level: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


@contextmanager
def capture_graph():
    """
    Context manager for capturing operations into a graph
    """
    global is_capturing, graph_operations, tensor_refs, next_tensor_id
    is_capturing = True
    graph_operations = []
    tensor_refs = {}
    next_tensor_id = 0

    class Graph:
        def __init__(self):
            self.operations = []

        def execute(self):
            """
            Execute the captured graph operations
            """
            print(f"Executing graph with {len(graph_operations)} operations:")

            for i, op in enumerate(graph_operations):
                func = op[0]
                print(f"  {i+1}. {func.__name__}:")

                if func == matmul:
                    a, b, device, tensor_id = op[1], op[2], op[3], op[4]
                    device = "cpu"  # Always use CPU for now

                    # Resolve tensor references
                    a_tensor = tensor_refs.get(a, a) if isinstance(a, str) else a
                    b_tensor = tensor_refs.get(b, b) if isinstance(b, str) else b

                    print(format_tensor_values(a_tensor, "Input A"))
                    print(format_tensor_values(b_tensor, "Input B"))

                    result = torch.matmul(a_tensor, b_tensor)
                    tensor_refs[tensor_id] = result
                    print(format_tensor_values(result, f"Output {tensor_id}"))

                elif func == silu:
                    x, device, tensor_id = op[1], op[2], op[3]
                    device = "cpu"  # Always use CPU for now

                    # Resolve tensor references
                    x_tensor = tensor_refs.get(x, x) if isinstance(x, str) else x

                    print(format_tensor_values(x_tensor, "Input X"))

                    result = F.silu(x_tensor)
                    tensor_refs[tensor_id] = result
                    print(format_tensor_values(result, f"Output {tensor_id}"))

                elif func == transform:
                    a, b, operation, device, tensor_id = (
                        op[1],
                        op[2],
                        op[3],
                        op[4],
                        op[5],
                    )
                    device = "cpu"  # Always use CPU for now

                    # Resolve tensor references
                    a_tensor = tensor_refs.get(a, a) if isinstance(a, str) else a
                    b_tensor = tensor_refs.get(b, b) if isinstance(b, str) else b

                    print(format_tensor_values(a_tensor, "Input A"))
                    print(format_tensor_values(b_tensor, "Input B"))
                    print(f"     Operation: {operation.__name__}")

                    result = operation(a_tensor, b_tensor)
                    tensor_refs[tensor_id] = result
                    print(format_tensor_values(result, f"Output {tensor_id}"))

                print()  # Add blank line between operations

            # Return the final output tensor (last one in the graph)
            final_tensor = list(tensor_refs.values())[-1] if tensor_refs else None
            return final_tensor

        def to_dot(self, filename=None):
            """
            Generate symbolic DOT graph representation and optionally save to file
            """
            dot_lines = [
                "digraph G {",
                "  rankdir=LR;",
                '  node [shape=box, style=filled, fontname="Arial"];',
                '  edge [fontname="Arial"];',
                "",
            ]

            # Create symbolic input nodes
            input_count = 0
            input_map = {}
            tensor_nodes = {}  # Map tensor IDs to their operation nodes

            # First pass: identify unique inputs and create symbolic names
            for op in graph_operations:
                func = op[0]
                if func == matmul:
                    a, b, device, tensor_id = op[1], op[2], op[3], op[4]
                    if not isinstance(a, str) and id(a) not in input_map:
                        input_map[id(a)] = f"input_{input_count}"
                        input_count += 1
                    if not isinstance(b, str) and id(b) not in input_map:
                        input_map[id(b)] = f"input_{input_count}"
                        input_count += 1
                elif func == silu:
                    x, device, tensor_id = op[1], op[2], op[3]
                    if not isinstance(x, str) and id(x) not in input_map:
                        input_map[id(x)] = f"input_{input_count}"
                        input_count += 1
                elif func == transform:
                    a, b, operation, device, tensor_id = (
                        op[1],
                        op[2],
                        op[3],
                        op[4],
                        op[5],
                    )
                    if not isinstance(a, str) and id(a) not in input_map:
                        input_map[id(a)] = f"input_{input_count}"
                        input_count += 1
                    if not isinstance(b, str) and id(b) not in input_map:
                        input_map[id(b)] = f"input_{input_count}"
                        input_count += 1

            # Add input nodes to graph
            for node_id, symbolic_name in input_map.items():
                dot_lines.append(
                    f'  "{symbolic_name}" [label="{symbolic_name}", fillcolor="lightblue"];'
                )

            # Add operation nodes and edges
            for i, op in enumerate(graph_operations):
                func = op[0]
                op_id = f"op_{i}"

                if func == matmul:
                    a, b, device, tensor_id = op[1], op[2], op[3], op[4]
                    dot_lines.append(
                        f'  "{op_id}" [label="MatMul\\n{tensor_id}", fillcolor="lightgreen"];'
                    )

                    # Add edges with symbolic names - only for actual input tensors, not tensor references
                    if not isinstance(a, str):
                        a_node = input_map.get(id(a), a)
                        dot_lines.append(f'  "{a_node}" -> "{op_id}";')
                    if not isinstance(b, str):
                        b_node = input_map.get(id(b), b)
                        dot_lines.append(f'  "{b_node}" -> "{op_id}";')

                    # Store this operation as the producer of tensor_id
                    tensor_nodes[tensor_id] = op_id

                elif func == silu:
                    x, device, tensor_id = op[1], op[2], op[3]
                    dot_lines.append(
                        f'  "{op_id}" [label="SiLU\\n{tensor_id}", fillcolor="lightyellow"];'
                    )

                    # Add edge with symbolic name - only for actual input tensors, not tensor references
                    if not isinstance(x, str):
                        x_node = input_map.get(id(x), x)
                        dot_lines.append(f'  "{x_node}" -> "{op_id}";')

                    # Store this operation as the producer of tensor_id
                    tensor_nodes[tensor_id] = op_id

                elif func == transform:
                    a, b, operation, device, tensor_id = (
                        op[1],
                        op[2],
                        op[3],
                        op[4],
                        op[5],
                    )
                    op_name = (
                        "Multiply"
                        if operation.__name__ == "<lambda>"
                        else operation.__name__.title()
                    )
                    dot_lines.append(
                        f'  "{op_id}" [label="{op_name}\\n{tensor_id}", fillcolor="lightcoral"];'
                    )

                    # Add edges with symbolic names - only for actual input tensors, not tensor references
                    if not isinstance(a, str):
                        a_node = input_map.get(id(a), a)
                        dot_lines.append(f'  "{a_node}" -> "{op_id}";')
                    if not isinstance(b, str):
                        b_node = input_map.get(id(b), b)
                        dot_lines.append(f'  "{b_node}" -> "{op_id}";')

                    # Store this operation as the producer of tensor_id
                    tensor_nodes[tensor_id] = op_id

            # Add tensor nodes (intermediate results) to the graph
            for tensor_id, producer_op in tensor_nodes.items():
                dot_lines.append(
                    f'  "{tensor_id}" [label="{tensor_id}", fillcolor="lightgray"];'
                )
                dot_lines.append(f'  "{producer_op}" -> "{tensor_id}";')

            # Add edges from tensor nodes to operations that consume them
            for i, op in enumerate(graph_operations):
                func = op[0]
                op_id = f"op_{i}"

                if func == matmul:
                    a, b, device, tensor_id = op[1], op[2], op[3], op[4]
                    # If a is a tensor reference, connect it to this operation
                    if isinstance(a, str) and a in tensor_nodes:
                        dot_lines.append(f'  "{a}" -> "{op_id}";')
                    # If b is a tensor reference, connect it to this operation
                    if isinstance(b, str) and b in tensor_nodes:
                        dot_lines.append(f'  "{b}" -> "{op_id}";')

                elif func == silu:
                    x, device, tensor_id = op[1], op[2], op[3]
                    # If x is a tensor reference, connect it to this operation
                    if isinstance(x, str) and x in tensor_nodes:
                        dot_lines.append(f'  "{x}" -> "{op_id}";')

                elif func == transform:
                    a, b, operation, device, tensor_id = (
                        op[1],
                        op[2],
                        op[3],
                        op[4],
                        op[5],
                    )
                    # If a is a tensor reference, connect it to this operation
                    if isinstance(a, str) and a in tensor_nodes:
                        dot_lines.append(f'  "{a}" -> "{op_id}";')
                    # If b is a tensor reference, connect it to this operation
                    if isinstance(b, str) and b in tensor_nodes:
                        dot_lines.append(f'  "{b}" -> "{op_id}";')

            dot_lines.append("}")
            dot_content = "\n".join(dot_lines)

            if filename:
                with open(filename, "w") as f:
                    f.write(dot_content)

            return dot_content

        def visualize(self, filename="graph.png"):
            """
            Generate DOT graph and create image file using dot command
            Automatically detects format from filename extension
            """
            import subprocess
            import os

            # Check if dot command exists
            try:
                subprocess.run(["dot", "-V"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"❌ 'dot' command not found. Please install Graphviz:")
                print(f"  macOS: brew install graphviz")
                print(f"  Ubuntu/Debian: sudo apt-get install graphviz")
                print(f"  Windows: Download from https://graphviz.org/download/")
                return None

            # Extract format from filename extension
            format = os.path.splitext(filename)[1][1:]  # Remove the dot

            # Generate DOT content
            dot_content = self.to_dot()

            # Create image using dot command
            try:
                result = subprocess.run(
                    ["dot", f"-T{format}", "-o", filename],
                    input=dot_content,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"✅ Graph visualization saved to {filename}")
                return filename
            except subprocess.CalledProcessError as e:
                print(f"❌ Error generating image: {e}")
                return None

    graph = Graph()

    try:
        yield graph
    finally:
        is_capturing = False
        graph.operations = graph_operations.copy()
