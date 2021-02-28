
#include <xaiengine.h>
#include <stdio.h>

/// Print the status of a core represented by the given tile, at the given coordinates.
void ACDC_print_tile_status(struct XAieGbl_Tile &tile) {
    int col = tile.ColId;
    int row = tile.RowId;
    u32 status, coreTimerLow, PC, LR, SP, locks, R0, R4;

    status = XAieGbl_Read32(tile.TileAddr + 0x032004);
    coreTimerLow = XAieGbl_Read32(tile.TileAddr + 0x0340F8);
    PC = XAieGbl_Read32(tile.TileAddr + 0x00030280);
    LR = XAieGbl_Read32(tile.TileAddr + 0x000302B0);
    SP = XAieGbl_Read32(tile.TileAddr + 0x000302A0);
    locks = XAieGbl_Read32(tile.TileAddr + 0x0001EF00);
    u32 trace_status = XAieGbl_Read32(tile.TileAddr + 0x000140D8);


    R0 = XAieGbl_Read32(tile.TileAddr + 0x00030000);
    R4 = XAieGbl_Read32(tile.TileAddr + 0x00030040);
    printf("Core [%d, %d] status is %08X, timer is %u, PC is %d, locks are %08X, LR is %08X, SP is %08X, R0 is %08X,R4 is %08X\n",col, row, status, coreTimerLow, PC, locks, LR, SP, R0, R4);
    printf("Core [%d, %d] trace status is %08X\n",col, row, trace_status);

    //printf("Check locks.\n");
    //locks = XAieGbl_Read32(tile.TileAddr + 0x0001EF00);
    for (int lock=0;lock<16;lock++) {
        u32 two_bits = (locks >> (lock*2)) & 0x3;
        if (two_bits) {
            printf("Lock %d: ", lock);
            u32 acquired = two_bits & 0x1;
            u32 value = two_bits & 0x2;
            if (acquired)
                printf("Acquired ");
            printf(value?"1":"0");
            printf("\n");
        }
    }
}