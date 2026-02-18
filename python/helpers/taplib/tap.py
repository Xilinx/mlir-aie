from __future__ import annotations

from copy import deepcopy
import numpy as np
import itertools
from typing import Sequence, Generator

from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)


class TensorAccessPattern:
    """
    A TensorAccessPattern represents a data access pattern applied to a tensor
    of a specific dimension. This is a base class meant to generically represent
    such as transformation using sizes, strides, and an offset.
    """

    _DTYPE = np.int32

    def __init__(
        self,
        tensor_dims: Sequence[int],
        offset: int = 0,
        sizes: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
    ):
        """
        An object representing an access pattern applied to a tensor.

        Args:
            tensor_dims (Sequence[int]): Dimensions of the tensor
            offset (int, optional): Offset into the tensor to begin the transformation. Defaults to 0.
            sizes (Sequence[int] | None, optional): Transformation sizes. Defaults to None.
            strides (Sequence[int] | None, optional): Transformation strides. Defaults to None.
        """
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        self._offset = validate_offset(offset, tensor_dims)

        if sizes is None:
            sizes = self._tensor_dims

        if strides is None:
            # Calculate contiguous strides
            strides = [1] * len(sizes)
            for i in range(len(sizes) - 2, -1, -1):
                strides[i] = strides[i + 1] * sizes[i + 1]

        self._sizes, self._strides = validate_and_clean_sizes_strides(sizes, strides)
        self._validate_bounds()

    def _validate_bounds(self):
        """
        Validate that the access pattern stays within the bounds of the tensor.
        """
        if not self._sizes:
            return

        max_index = self._offset
        for size, stride in zip(self._sizes, self._strides):
            if size > 0:
                max_index += (size - 1) * stride

        total_size = np.prod(self._tensor_dims)
        if max_index >= total_size:
            raise ValueError(
                f"Access pattern goes out of bounds: max_index={max_index}, total_size={total_size}"
            )

    @property
    def tensor_dims(self) -> Sequence[int]:
        """
        A copy of the dimensions of the tensor

        Returns:
            Sequence[int]: Tensor dimensions
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._tensor_dims)

    @property
    def offset(self) -> int:
        """
        Returns the offset into the tensor

        Returns:
            int: offset
        """
        return self._offset

    @property
    def sizes(self) -> Sequence[int]:
        """
        A copy of the access pattern sizes

        Returns:
            Sequence[int]: Transformation sizes
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._sizes)

    @property
    def strides(self) -> Sequence[int]:
        """
        A copy of the access pattern strides

        Returns:
            Sequence[int]: Trsnformation strides
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._strides)

    @property
    def transformation_dims(self) -> Sequence[tuple[int, int]]:
        """
        The access pattern represented as a sequence of (size, stride) tuples

        Returns:
            Sequence[tuple[int, int]]: Transformation dimensions
        """
        return list(zip(self._sizes, self._strides))

    def accesses(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the access_order and access_count arrays.

        The access_order ndarray sequentially counts access to elements in the
        tensor. If an element is accessed more than once, only the last count is reflected.

        The access_count ndarray contains the number of times each element is
        accessed by the tensor access pattern.

        Returns:
            tuple[np.ndarray, np.ndarray]: access_order, access_count
        """
        return self._calculate_accesses(calc_order=True, calc_count=True)

    def access_order(self) -> np.ndarray:
        """
        The access_order ndarray sequentially counts access to elements in the
        tensor. If an element is accessed more than once, only the last count is reflected.

        Returns:
            np.ndarray: access_order
        """
        access_order_tensor, _ = self._calculate_accesses(
            calc_order=True, calc_count=False
        )
        return access_order_tensor

    def access_count(self) -> np.ndarray:
        """
        The access_count ndarray contains the number of times each element is
        accessed by the tensor access pattern.

        Returns:
            np.ndarray: access_count
        """
        _, access_count_tensor = self._calculate_accesses(
            calc_order=False, calc_count=True
        )
        return access_count_tensor

    def _calculate_accesses(
        self, calc_order: bool, calc_count: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        # This is an internal method for calculating both the access_order and access_count
        # arrays. If needed, it will create both at once to avoid looping through the tensor
        # more than necessary.

        # TODO: should access order be a list of lists instead of generate two separate tensors?
        # TODO: for performance, should cache and return copies? Or just cache?
        if not calc_order and not calc_count:
            raise ValueError("Must select calc_order, calc_count, or both")

        # Initialize access order and count maps; we create them as flat arrays
        total_elems = np.prod(self._tensor_dims)
        access_order_tensor = None
        if calc_order:
            access_order_tensor = np.full(total_elems, -1, dtype=self._DTYPE)
            access_count = 0
        access_count_tensor = None
        if calc_count:
            access_count_tensor = np.full(total_elems, 0, dtype=self._DTYPE)

        # Get an iterator for the access indices
        access_idx_generator = self.access_generator()

        for access_idx in access_idx_generator:
            # Count the accesses
            if calc_count:
                access_count_tensor[access_idx] += 1
            # Enumerate the accesses
            if calc_order:
                access_order_tensor[access_idx] = access_count
                access_count += 1

        # Reshape to match tensor type since we created them initially as flat arrays
        if calc_order:
            access_order_tensor = access_order_tensor.reshape(self._tensor_dims)
        if calc_count:
            access_count_tensor = access_count_tensor.reshape(self._tensor_dims)
        return access_order_tensor, access_count_tensor

    def access_generator(self) -> Generator[int, None, None]:
        """This function returns an iterator that returns the access index
        into the flattened tensor that this access pattern represents. This can
        be used to calculate the access count or to enumerate accesses.

        Yields:
            int: The next access index
        """
        total_elems = np.prod(self._tensor_dims)

        # Use itertools.product to collapse len(sizes) nested forloop into one forloop
        for dims in itertools.product(*[range(0, n) for n in self._sizes]):
            yield (
                self._offset + np.sum(np.multiply(dims, self._strides))
            ) % total_elems

    def compare_access_orders(self, other: TensorAccessPattern) -> bool:
        """
        This function creates an alternative way to compare access patterns.
        Sometimes access patterns with different sizes/strides are functionally equivalent;
        to detect functional equivalency, this function uses iterators produced by
        access_generator() to compare the access patterns. This is more performant than
        comparing the numpy array access_order or access_count tensors.

        Args:
            other (TensorAccessPattern): The TensorAccessPattern to compare to

        Raises:
            ValueError: other must be of type TensorAccessPattern

        Returns:
            bool: True if the TensorAccessPatterns are functionally equivalent; false otherwise.
        """
        # This function compares using access generators, which is more performant
        # than actually generating the access order or access count tensors.
        if not isinstance(other, TensorAccessPattern):
            raise ValueError(
                "Can only compare access order against another TensorAccessPattern"
            )
        my_generator = self.access_generator()
        other_generator = other.access_generator()
        return all(
            my_idx == other_idx
            for my_idx, other_idx in itertools.zip_longest(
                my_generator, other_generator
            )
        )

    def compose(self, other: TensorAccessPattern) -> TensorAccessPattern:
        """
        Compose this access pattern with another.
        The resulting pattern represents applying 'other' to the view created by 'self'.

        Args:
            other (TensorAccessPattern): The outer access pattern.

        Returns:
            TensorAccessPattern: The composed access pattern.
        """
        if tuple(other.tensor_dims) != tuple(self.sizes):
            raise ValueError(
                f"Dimension mismatch: self.sizes={self.sizes}, other.tensor_dims={other.tensor_dims}"
            )

        # Calculate new offset
        new_offset = self._map_linear_index(other.offset)

        # Calculate new strides
        new_strides = []
        for s in other.strides:
            mapped_stride = self._map_linear_stride(s)
            new_strides.append(mapped_stride)

        return TensorAccessPattern(
            self.tensor_dims, new_offset, other.sizes, new_strides
        )

    def permute(self, dims: Sequence[int]) -> TensorAccessPattern:
        """
        Permute the dimensions of the view.

        Args:
            dims (Sequence[int]): The new order of dimensions.

        Returns:
            TensorAccessPattern: The permuted access pattern.
        """
        if set(dims) != set(range(len(self.sizes))):
            raise ValueError("Invalid permutation")

        new_sizes = [self.sizes[i] for i in dims]
        new_strides = [self.strides[i] for i in dims]
        return TensorAccessPattern(
            self.tensor_dims, self.offset, new_sizes, new_strides
        )

    def split_dim(self, dim: int, outer_size: int) -> TensorAccessPattern:
        """
        Split a dimension into two dimensions.

        Args:
            dim (int): The dimension to split.
            outer_size (int): The size of the outer dimension.

        Returns:
            TensorAccessPattern: The access pattern with the split dimension.
        """
        if dim < 0 or dim >= len(self.sizes):
            raise ValueError(f"Invalid dimension {dim}")
        if outer_size <= 0:
            raise ValueError(f"Outer size must be positive, got {outer_size}")
        if self.sizes[dim] % outer_size != 0:
            raise ValueError(
                f"Dimension size {self.sizes[dim]} not divisible by {outer_size}"
            )

        inner_size = self.sizes[dim] // outer_size
        old_stride = self.strides[dim]

        new_sizes = list(self.sizes)
        new_sizes[dim : dim + 1] = [outer_size, inner_size]

        new_strides = list(self.strides)
        new_strides[dim : dim + 1] = [old_stride * inner_size, old_stride]

        return TensorAccessPattern(
            self.tensor_dims, self.offset, new_sizes, new_strides
        )

    def tile(self, tile_sizes: Sequence[int]) -> TensorAccessPattern:
        """
        Tile the access pattern by splitting dimensions and grouping outer/inner dimensions.

        Args:
            tile_sizes (Sequence[int]): The size of the tile for each dimension.

        Returns:
            TensorAccessPattern: The tiled access pattern with dimensions
            (dim0//tile0, dim1//tile1, ..., tile0, tile1, ...).
        """
        if len(tile_sizes) != len(self.sizes):
            raise ValueError("tile_sizes must match number of dimensions")

        tap = self
        # Split each dimension from last to first to preserve indices
        for i in range(len(self.sizes) - 1, -1, -1):
            if tile_sizes[i] <= 0:
                raise ValueError("Tile sizes must be positive")
            tap = tap.split_dim(i, self.sizes[i] // tile_sizes[i])

        # Now we have (d0_out, d0_in, d1_out, d1_in, ...)
        # We want (d0_out, d1_out, ..., d0_in, d1_in, ...)

        n = len(self.sizes)  # Original rank
        perm = []
        for i in range(n):
            perm.append(2 * i)  # Outer dims
        for i in range(n):
            perm.append(2 * i + 1)  # Inner dims

        return tap.permute(perm)

    def slice(
        self, dim: int, start: int, length: int, step: int = 1
    ) -> TensorAccessPattern:
        """
        Slice a dimension.

        Args:
            dim (int): The dimension to slice.
            start (int): The start index.
            length (int): The length of the slice.
            step (int, optional): The step size. Defaults to 1.

        Returns:
            TensorAccessPattern: The sliced access pattern.
        """
        if dim < 0 or dim >= len(self.sizes):
            raise ValueError(f"Invalid dimension {dim}")
        if length <= 0:
            raise ValueError(f"Slice length must be positive, got {length}")
        if step <= 0:
            raise ValueError(f"Slice step must be positive, got {step}")

        # Check bounds: start + (length-1)*step must be < size
        max_idx = start + (length - 1) * step
        if start < 0 or max_idx >= self.sizes[dim]:
            raise ValueError(
                f"Invalid slice {start}:{max_idx+1}:{step} for dimension {dim} of size {self.sizes[dim]}"
            )

        new_offset = self.offset + start * self.strides[dim]
        new_sizes = list(self.sizes)
        new_sizes[dim] = length

        new_strides = list(self.strides)
        new_strides[dim] *= step

        return TensorAccessPattern(self.tensor_dims, new_offset, new_sizes, new_strides)

    def subview(
        self,
        offsets: Sequence[int],
        sizes: Sequence[int],
        steps: Sequence[int] | None = None,
    ) -> TensorAccessPattern:
        """
        Create a subview of the access pattern.

        Args:
            offsets (Sequence[int]): The offsets for each dimension.
            sizes (Sequence[int]): The sizes for each dimension.
            steps (Sequence[int] | None, optional): The steps for each dimension. Defaults to None (1).

        Returns:
            TensorAccessPattern: The subview access pattern.
        """
        if len(offsets) != len(self.sizes) or len(sizes) != len(self.sizes):
            raise ValueError("Rank mismatch")
        if steps is None:
            steps = [1] * len(self.sizes)
        elif len(steps) != len(self.sizes):
            raise ValueError("Rank mismatch for steps")

        tap = self
        for i in range(len(self.sizes)):
            tap = tap.slice(i, offsets[i], sizes[i], steps[i])
        return tap

    def unsqueeze(self, dim: int) -> TensorAccessPattern:
        """
        Insert a dimension of size 1 at the specified position.

        Args:
            dim (int): The index at which to insert the new dimension.

        Returns:
            TensorAccessPattern: The access pattern with the new dimension.
        """
        if dim < 0 or dim > len(self.sizes):
            raise ValueError(f"Invalid dimension {dim}")

        new_sizes = list(self.sizes)
        new_sizes.insert(dim, 1)

        new_strides = list(self.strides)
        new_strides.insert(dim, 0)  # Stride 0 for size 1

        return TensorAccessPattern(
            self.tensor_dims, self.offset, new_sizes, new_strides
        )

    def broadcast(self, dim: int, size: int) -> TensorAccessPattern:
        """
        Broadcast a dimension of size 1 to a larger size.

        Args:
            dim (int): The dimension to broadcast.
            size (int): The new size.

        Returns:
            TensorAccessPattern: The broadcasted access pattern.
        """
        if dim < 0 or dim >= len(self.sizes):
            raise ValueError(f"Invalid dimension {dim}")
        if self.sizes[dim] != 1:
            raise ValueError(
                f"Dimension {dim} must be size 1 to broadcast, got {self.sizes[dim]}"
            )
        if size <= 0:
            raise ValueError(f"Broadcast size must be positive, got {size}")

        new_sizes = list(self.sizes)
        new_sizes[dim] = size

        # Stride remains 0 (or whatever it was, but usually 0 for size 1 broadcast)
        # If stride was not 0, broadcasting implies repeating the element, so stride should be 0?
        # But if we broadcast size 1, we repeat the single element.
        # So stride effectively becomes 0 for the new dimension?
        # If stride was S, and size was 1. We access index 0.
        # If we broadcast to size N. We want to access index 0 N times.
        # So stride must be 0.

        new_strides = list(self.strides)
        new_strides[dim] = 0

        return TensorAccessPattern(
            self.tensor_dims, self.offset, new_sizes, new_strides
        )

    def squeeze(self, dim: int | None = None) -> TensorAccessPattern:
        """
        Remove a dimension of size 1.

        Args:
            dim (int | None): The dimension to remove. If None, remove all dimensions of size 1.

        Returns:
            TensorAccessPattern: The squeezed access pattern.
        """
        if dim is not None:
            if dim < 0 or dim >= len(self.sizes):
                raise ValueError(f"Invalid dimension {dim}")
            if self.sizes[dim] != 1:
                raise ValueError(
                    f"Cannot squeeze dimension {dim} of size {self.sizes[dim]}"
                )
            new_sizes = list(self.sizes)
            del new_sizes[dim]
            new_strides = list(self.strides)
            del new_strides[dim]
        else:
            new_sizes = []
            new_strides = []
            for s, st in zip(self.sizes, self.strides):
                if s != 1:
                    new_sizes.append(s)
                    new_strides.append(st)
            # If we squeezed everything (scalar), we might want to keep it as 0-rank or 1-rank?
            # TensorAccessPattern supports empty sizes (rank 0)?
            # __init__ checks len(sizes) > 0 if not None.
            # So we must keep at least one dimension?
            # validate_and_clean_sizes_strides checks len > 0.
            # So we should keep one if empty.
            if not new_sizes:
                new_sizes = [1]
                new_strides = [0]  # Stride doesn't matter for size 1

        return TensorAccessPattern(
            self.tensor_dims, self.offset, new_sizes, new_strides
        )

    def resize(self, new_sizes: Sequence[int]) -> TensorAccessPattern:
        """
        Resize the access pattern by changing the sizes of dimensions.
        This is equivalent to slicing from 0 with the new sizes.
        Strides are preserved.

        Args:
            new_sizes (Sequence[int]): The new sizes for each dimension.

        Returns:
            TensorAccessPattern: The resized access pattern.
        """
        if len(new_sizes) != len(self.sizes):
            raise ValueError("Rank mismatch")

        return self.subview([0] * len(self.sizes), new_sizes)

    def reshape(self, new_shape: Sequence[int]) -> TensorAccessPattern:
        """
        Reshape the access pattern.
        This only works if the current pattern is contiguous in memory
        and can be viewed as the new shape with constant strides.

        Args:
            new_shape (Sequence[int]): The new shape.

        Returns:
            TensorAccessPattern: The reshaped access pattern.
        """
        if np.prod(new_shape) != np.prod(self.sizes):
            raise ValueError(
                f"Total size mismatch: {np.prod(new_shape)} != {np.prod(self.sizes)}"
            )

        # Check if contiguous
        # We can check if we can linearize to (total_size,) with some stride S.
        # But simpler: if we are identity-like (contiguous row-major), we can reshape easily.
        # Or if we can untile to flat?
        # Let's try to construct strides for new_shape assuming row-major layout of the underlying data
        # as seen by the current view.
        # This is hard if current view is not contiguous.
        # But for identity((N,)), we can reshape.
        # If we assume the underlying data is contiguous row-major with respect to current view?
        # No, TAP describes view on T_orig.
        # If T_orig is contiguous, and TAP is identity, then TAP is contiguous.
        # If TAP is contiguous, we can calculate new strides based on T_orig strides?
        # If TAP covers a contiguous range of T_orig, we can reshape.
        # For now, let's just support reshaping if it's equivalent to tiling/untiling.
        # But tile/untile is safer.
        # However, for identity((4,)), reshape((2, 2)) is just tile((2,)).
        # So maybe we don't need reshape if we have tile?
        # But tile adds dimensions. Reshape replaces them.
        # identity((4,)).tile((2,)) -> (2, 2).
        # So tile IS reshape for 1D?
        # Yes.
        # So we can use tile.

        # I will skip reshape for now to avoid complexity and potential bugs.
        # subview is enough for the user request.
        raise NotImplementedError("Reshape not implemented yet")

    def untile(
        self, tile_sizes: Sequence[int], merge: bool = True
    ) -> TensorAccessPattern:
        """
        Untile the access pattern.
        This is the reverse of tile().
        It expects dimensions to be (dim0//tile0, dim1//tile1, ..., tile0, tile1, ...).
        It permutes them to (dim0//tile0, tile0, dim1//tile1, tile1, ...)
        and optionally merges them to (dim0, dim1, ...).

        Args:
            tile_sizes (Sequence[int]): The size of the tiles.
            merge (bool, optional): Whether to merge the split dimensions. Defaults to True.

        Returns:
            TensorAccessPattern: The untiled access pattern.
        """
        n = len(tile_sizes)
        if len(self.sizes) != 2 * n:
            raise ValueError(
                f"Expected {2*n} dimensions for untiling with {n} tile sizes, got {len(self.sizes)}"
            )

        # Permute from (out0, out1, ..., in0, in1, ...) to (out0, in0, out1, in1, ...)
        perm = []
        for i in range(n):
            perm.append(i)  # out_i
            perm.append(n + i)  # in_i

        tap = self.permute(perm)

        if not merge:
            return tap

        # Merge dimensions
        new_sizes = []
        new_strides = []

        for i in range(n):
            out_size = tap.sizes[2 * i]
            in_size = tap.sizes[2 * i + 1]
            out_stride = tap.strides[2 * i]
            in_stride = tap.strides[2 * i + 1]

            if out_stride != in_size * in_stride:
                raise ValueError(
                    f"Cannot merge dimension {i}: strides do not match (out_stride={out_stride}, in_size={in_size}, in_stride={in_stride})"
                )

            new_sizes.append(out_size * in_size)
            new_strides.append(in_stride)

        return TensorAccessPattern(
            self.tensor_dims, self.offset, new_sizes, new_strides
        )

    def _map_linear_index(self, linear_index: int) -> int:
        coords = self._decompose_index(linear_index, self.sizes)
        return self._offset + np.sum(np.multiply(coords, self._strides))

    def _map_linear_stride(self, linear_stride: int) -> int:
        coords = self._decompose_index(linear_stride, self.sizes)
        return np.sum(np.multiply(coords, self._strides))

    def _decompose_index(self, index: int, dims: Sequence[int]) -> Sequence[int]:
        coords = []
        contiguous_strides = self._get_contiguous_strides(dims)

        rem = index
        for i in range(len(dims)):
            val = rem // contiguous_strides[i]
            rem = rem % contiguous_strides[i]
            coords.append(val)
        return coords

    def _linearize_index(self, coords: Sequence[int], dims: Sequence[int]) -> int:
        contiguous_strides = self._get_contiguous_strides(dims)
        return np.sum(np.multiply(coords, contiguous_strides))

    def _get_contiguous_strides(self, dims: Sequence[int]) -> Sequence[int]:
        strides = [1] * len(dims)
        for i in range(len(dims) - 2, -1, -1):
            strides[i] = strides[i + 1] * dims[i + 1]
        return strides

    def _solve_indices(self, target: int) -> Sequence[int] | None:
        # Solve offset + sum(x_i * stride_i) = target for x_i in [0, size_i)
        # Greedy approach with sorting strides
        dims = []
        for i in range(len(self.strides)):
            dims.append((self.strides[i], self.sizes[i], i))

        # Sort by stride descending
        dims.sort(key=lambda x: abs(x[0]), reverse=True)

        current = target - self.offset
        solution = [0] * len(self.sizes)

        for stride, size, idx in dims:
            if stride == 0:
                continue

            # If stride is negative, we need to handle it carefully
            # But usually strides are positive in this context.
            # If stride is negative, we might need to subtract?
            # Let's assume positive strides for greedy.
            # If mixed, greedy is hard.

            count = current // stride

            # Check bounds
            if count >= size:
                count = size - 1
            elif count < 0:
                # This shouldn't happen if everything is positive and target >= offset
                count = 0

            # We need to check if taking 'count' is valid.
            # For standard tilings/permutations, greedy works.

            solution[idx] = count
            current -= count * stride

        if current != 0:
            return None

        return tuple(solution)

    def visualize(
        self,
        show_arrows: bool | None = None,
        title: str | None = None,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        """Visualize the TensorAccessPattern using a graph.

        Args:
            show_arrows (bool | None, optional): Display arrows between sequentially accessed elements. Defaults to None.
            title (str | None, optional): Title of the produced graph. Defaults to None.
            file_path (str | None, optional): Path to save the graph at; if none, it is not saved. Defaults to None.
            show_plot (bool, optional): Show the plot (this is useful for Jupyter notebooks). Defaults to True.
            plot_access_count (bool, optional): Plot the access count in addition to the access order. Defaults to False.

        Raises:
            NotImplementedError: This function is not implemented for all dimensions.
        """
        from .visualization2d import visualize_from_accesses

        if len(self._tensor_dims) != 2:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )
        if plot_access_count:
            access_order, access_count = self.accesses()
        else:
            access_count = None
            access_order = self.access_order()
        if title is None:
            title = str(self)
        visualize_from_accesses(
            access_order,
            access_count,
            title=title,
            show_arrows=show_arrows,
            file_path=file_path,
            show_plot=show_plot,
        )

    def __str__(self) -> str:
        return f"TensorAccessPattern({self.tensor_dims} offset={self._offset}, sizes={self._sizes}, strides={self._strides})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self._tensor_dims == other._tensor_dims
                and self._offset == other._offset
                and self._sizes == other._sizes
                and self._strides == other._strides
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def tile_sequence(
        self,
        tile_dims: Sequence[int],
        step_dims: Sequence[int] | None = None,
        repeat_dims: Sequence[int] | None = None,
        dim_order: Sequence[int] | None = None,
        tile_dim_order: Sequence[int] | None = None,
        repeat_dim_order: Sequence[int] | None = None,
        pattern_repeat: int = 1,
    ) -> "TensorAccessSequence":
        """
        Generate a sequence of access patterns by tiling the current pattern.

        Args:
            tile_dims: Dimensions of the tile.
            step_dims: Step size in units of tiles for each dimension. Defaults to 1 (contiguous).
            repeat_dims: Number of tiles to group together in each dimension. Defaults to 1.
            dim_order: Order of iteration over dimensions. Defaults to range(rank) (row-major-like).
            tile_dim_order: Order of dimensions within the tile. Defaults to range(rank) (row-major-like).
            repeat_dim_order: Order of dimensions for the tile group. Defaults to range(rank) (row-major-like).
            pattern_repeat: Number of times to repeat the pattern within each step.

        Returns:
            TensorAccessSequence: The sequence of access patterns.
        """
        from .tas import TensorAccessSequence
        from .utils import ceildiv

        rank = len(self.sizes)
        if len(tile_dims) != rank:
            raise ValueError("Rank mismatch between tensor_dims and tile_dims")

        if step_dims is None:
            step_dims = [1] * rank
        if repeat_dims is None:
            repeat_dims = [1] * rank
        if dim_order is None:
            dim_order = list(range(rank))
        if tile_dim_order is None:
            tile_dim_order = list(range(rank))
        if repeat_dim_order is None:
            repeat_dim_order = list(range(rank))

        if (
            len(step_dims) != rank
            or len(repeat_dims) != rank
            or len(dim_order) != rank
            or len(tile_dim_order) != rank
            or len(repeat_dim_order) != rank
        ):
            raise ValueError("Rank mismatch in arguments")

        # Calculate steps per dimension
        steps_per_dim = []
        for i in range(rank):
            tiles = ceildiv(self.sizes[i], tile_dims[i])
            block_size_tiles = step_dims[i] * repeat_dims[i]

            if block_size_tiles > 0:
                num_full_blocks = tiles // block_size_tiles
                steps_from_blocks = num_full_blocks * step_dims[i]
            else:
                num_full_blocks = 0
                steps_from_blocks = 0

            leftover_tiles = tiles - (block_size_tiles * num_full_blocks)
            leftover_repeat_dim = leftover_tiles // step_dims[i]
            leftover_block_dim_in_tiles = step_dims[i] * leftover_repeat_dim
            if leftover_block_dim_in_tiles > 0:
                num_leftover_blocks = leftover_tiles // leftover_block_dim_in_tiles
                steps_from_leftover_blocks = num_leftover_blocks * step_dims[i]
            else:
                steps_from_leftover_blocks = leftover_tiles

            steps_per_dim.append(steps_from_blocks + steps_from_leftover_blocks)

        num_steps = np.prod(steps_per_dim)

        def get_step_indices(step):
            step_indices = [0] * rank
            rem = step
            for dim in reversed(dim_order):
                count = steps_per_dim[dim]
                val = rem % count
                rem = rem // count
                step_indices[dim] = val
            return step_indices

        def get_tap(step):
            indices = get_step_indices(step)

            # Construct TAP
            tap = self.tile(tile_dims)  # (dim0_out, dim0_in, dim1_out, dim1_in, ...)

            # Calculate available tiles in each dim
            total_tiles = [ceildiv(self.sizes[i], tile_dims[i]) for i in range(rank)]
            start_tiles = []
            for i in range(rank):
                block_idx = indices[i] // step_dims[i]
                interleave_idx = indices[i] % step_dims[i]
                start_tiles.append(
                    block_idx * step_dims[i] * repeat_dims[i] + interleave_idx
                )

            avail_tiles = [total_tiles[i] - start_tiles[i] for i in range(rank)]

            actual_repeats = []
            actual_steps = []

            for i in range(rank):
                rep = min(repeat_dims[i], ceildiv(avail_tiles[i], step_dims[i]))
                actual_repeats.append(rep)
                step = step_dims[i]
                if step > avail_tiles[i]:
                    step = 1
                actual_steps.append(step)

            # Slice
            # tile() produces (out0, out1, ..., in0, in1, ...)
            # Wait, my tile implementation produces (out0, in0, out1, in1, ...) ?
            # Let's check tile implementation in this file.
            # perm.append(2 * i) -> 0, 2, 4...
            # perm.append(2 * i + 1) -> 1, 3, 5...
            # So (out0, out1, ..., in0, in1, ...).
            # So outer dims are 0 to rank-1.
            # Inner dims are rank to 2*rank-1.

            for i in range(rank):
                # Slice outer dimension i (which is at index i)
                # Start index is 0 relative to the offset handled by offset_fn?
                # No, offset_fn handles offset in original tensor.
                # But here we are constructing a TAP that represents the view relative to that offset.
                # So we slice from 0.
                tap = tap.slice(i, 0, actual_repeats[i], actual_steps[i])

            # Handle tile_dim_order (permute inner dims)
            # Inner dims are at indices [rank, rank+1, ..., 2*rank-1]
            # We want to permute them according to tile_dim_order.
            # e.g. if tile_dim_order=[1, 0], we swap rank and rank+1.
            if tile_dim_order != list(range(rank)):
                perm = list(range(rank)) + [rank + i for i in tile_dim_order]
                tap = tap.permute(perm)

            # Handle repeat_dim_order (permute outer dims)
            # Outer dims are at indices [0, 1, ..., rank-1]
            if repeat_dim_order != list(range(rank)):
                perm = [repeat_dim_order[i] for i in range(rank)] + list(
                    range(rank, 2 * rank)
                )
                tap = tap.permute(perm)

            # Handle pattern repeat
            if pattern_repeat > 1:
                tap = tap.unsqueeze(0)
                tap = tap.broadcast(0, pattern_repeat)

            # Squeeze size 1 dimensions to fit in 4 dims if possible
            # But only if we have > 4 dims?
            # Or always squeeze?
            # TensorTiler2D logic tries to fit in 4 dims.
            # If we have > 4 dims, we MUST squeeze.
            # If we have <= 4 dims, squeezing is optional but good for cleanup.
            # However, squeezing might change the interpretation if the user expects specific rank.
            # But for DMA BDs, we just want to fit in 4 dims.
            # So let's squeeze if > 4 dims.
            if len(tap.sizes) > 4:
                tap = tap.squeeze()

            return tap

        def offset_fn(step, _):
            indices = get_step_indices(step)
            offset = 0
            for i in range(rank):
                # Offset in tiles
                block_idx = indices[i] // step_dims[i]
                interleave_idx = indices[i] % step_dims[i]
                tile_idx = block_idx * step_dims[i] * repeat_dims[i] + interleave_idx

                # Offset in elements = tile_idx * tile_size * stride(dim i)
                # Stride of dim i in original tensor (self)
                offset += tile_idx * tile_dims[i] * self.strides[i]
            return self.offset + offset

        return TensorAccessSequence(
            self.tensor_dims,
            num_steps,
            sizes_fn=lambda s, _: get_tap(s).sizes,
            strides_fn=lambda s, _: get_tap(s).strides,
            offset_fn=offset_fn,
        )
