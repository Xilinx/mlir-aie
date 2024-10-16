class SubtileLoopSyncPolicy:
    """Sync at end of every subtile loop iteration. If no subtile loops, no syncs will be inserted"""

    pass


class TileLoopSyncPolicy:
    """Sync at end of every tile loop iteration. If no tile loops, no syncs will be inserted"""

    pass


class NoSyncPolicy:
    """Do not insert any syncs. This is useful if manually adding syncs."""

    pass


class SingleSyncPolicy:
    """Add one sync at the end of the all operations."""

    pass


class NSyncPolicy:
    """Sync every n ops per shim tile. Defaults to one if not given"""

    def __init__(self, num_ops: int = 1):
        assert num_ops >= 1
        self.num_ops = num_ops


DMASyncPolicy = (
    SubtileLoopSyncPolicy
    | TileLoopSyncPolicy
    | NoSyncPolicy
    | SingleSyncPolicy
    | NSyncPolicy
)
