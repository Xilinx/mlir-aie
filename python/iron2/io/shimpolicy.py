class SingleShimPolicy:
    pass


class DistributeShimPolicy:
    def __init__(
        self, num_shim_tiles: int | None = None, chunk_size: int | None = None
    ):
        if num_shim_tiles:
            assert num_shim_tiles > 0
        if chunk_size:
            assert chunk_size > 0
        self.num_shim_tiles = num_shim_tiles
        self.chunk_size = chunk_size


ShimPlacementPolicy = SingleShimPolicy | DistributeShimPolicy
