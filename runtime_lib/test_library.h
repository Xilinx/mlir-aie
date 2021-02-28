#include <xaiengine.h>

/// Dump the contents of the memory associated with the given tile.
void ACDC_dump_tile_memory(struct XAieGbl_Tile &tile);

/// Clear the contents of the memory associated with the given tile.
void ACDC_clear_tile_memory(struct XAieGbl_Tile &tile);

/// Print the status of a dma represented by the given tile.
void ACDC_print_dma_status(struct XAieGbl_Tile &tile);

/// Print the status of a core represented by the given tile.
void ACDC_print_tile_status(struct XAieGbl_Tile &tile);