from .phys.tile import Tile, PlacementTile


class Placeable:
    def __init__(self, tile: PlacementTile):
        self._tile = tile

    def place(self, tile: Tile) -> None:
        if isinstance(self._tile, Tile):
            raise AlreadyPlacedError(self.__class__, self._tile, tile)
        self._tile = tile

    @property
    def tile(self) -> PlacementTile:
        return self._tile


class AlreadyPlacedError(Exception):
    def __init__(self, cls, current_tile: Tile, new_tile: Tile):
        self.message = (
            f"{cls} already placed at {current_tile}; cannot place at {new_tile}"
        )
        super().__init__(self.message)
