import ctypes
from ctypes import pointer

from cdo_driver import (
    startCDOFileStream,
    configureHeader,
    endCurrentCDOFileStream,
    String,
    FileHeader,
    Little_Endian,
    setEndianness,
)
from xaiengine import (
    XAie_CoreReset,
    XAie_CoreUnreset,
    XAie_LockSetValue,
    XAie_DmaDescInit,
    XAie_DmaSetLock,
    XAie_DevInst,
    XAie_LocType,
    XAie_Config,
    XAie_PartitionProp,
    XAie_SetupPartitionConfig,
    XAie_CfgInitialize,
    XAie_UpdateNpiAddr,
    XAie_ErrorHandlingInit,
    XAie_Lock,
    XAie_DmaDesc,
    XAie_DmaDescInit,
    XAie_DmaSetLock,
    XAie_DmaSetAddrLen,
    XAie_DmaSetNextBd,
    XAie_DmaEnableBd,
    XAie_DmaWriteBd,
    XAie_DmaChannelPushBdToQueue,
    XAie_DmaChannelEnable,
    DMA_S2MM,
    DMA_MM2S,
    XAie_StrmConnCctEnable,
    XAie_DmaChannelDesc,
    XAie_DmaChannelDescInit,
    XAie_DmaChannelSetControllerId,
    XAie_DmaWriteChannel,
    XAie_AieToPlIntfEnable,
    XAie_EnableShimDmaToAieStrmPort,
    XAie_EnableAieToShimDmaStrmPort,
    NORTH,
    SOUTH,
    CTRL,
    DMA,
    PLIF_WIDTH_32,
    XAie_CoreEnable,
)

XAie_TileLoc = lambda col, row: XAie_LocType(row, col)
XAie_LockInit = XAie_Lock

setEndianness(Little_Endian)

HW_GEN = XAIE_DEV_GEN_AIEML = 2
XAIE_NUM_ROWS = 6
XAIE_NUM_COLS = 5
XAIE_BASE_ADDR = 0x40000000
XAIE_COL_SHIFT = 25
XAIE_ROW_SHIFT = 20
XAIE_SHIM_ROW = 0
XAIE_MEM_TILE_ROW_START = 1
XAIE_MEM_TILE_NUM_ROWS = 1
XAIE_AIE_TILE_ROW_START = 2
XAIE_AIE_TILE_NUM_ROWS = 4
XAIE_PARTITION_BASE_ADDR = 0x0

DevInst = XAie_DevInst()

PartProp = XAie_PartitionProp()
ConfigPtr = XAie_Config(
    HW_GEN,
    XAIE_BASE_ADDR,
    XAIE_COL_SHIFT,
    XAIE_ROW_SHIFT,
    XAIE_NUM_COLS,
    XAIE_NUM_ROWS,
    XAIE_SHIM_ROW,
    XAIE_MEM_TILE_ROW_START,
    XAIE_MEM_TILE_NUM_ROWS,
    XAIE_AIE_TILE_ROW_START,
    XAIE_AIE_TILE_NUM_ROWS,
)

XAie_SetupPartitionConfig(DevInst, XAIE_PARTITION_BASE_ADDR, 1, 1)
XAie_CfgInitialize(DevInst, ConfigPtr)
XAie_UpdateNpiAddr(DevInst, 0x0)

print("START: Error Handling Configuration")
startCDOFileStream(String(b"aie_cdo_error_handling.bin"))
FileHeader()
XAie_ErrorHandlingInit(DevInst)
configureHeader()
endCurrentCDOFileStream()

#   // aie_cdo_elfs.bin
#   const std::string elfsCDOFilePath = workDirPath + "aie_cdo_elfs.bin";
#   if (AXIdebug)
#     std::cout << "START: AIE ELF Configuration\n";
#   startCDOFileStream(elfsCDOFilePath.c_str());
#   FileHeader();
#   addAieElfsToCDO(workDirPath);
#   configureHeader();
#   endCurrentCDOFileStream();
#   if (AXIdebug)
#     std::cout << "DONE: AIE ELF Configuration\n\n";

print("Initial Configuration (SHIM and AIE Array)")
startCDOFileStream(String(b"aie_cdo_init.bin"))
FileHeader()

tile = XAie_TileLoc(0, 2)
XAie_CoreReset(DevInst, tile)
XAie_CoreUnreset(DevInst, tile)

for i in range(16):
    XAie_LockSetValue(DevInst, tile, XAie_LockInit(i, 0))
XAie_LockSetValue(DevInst, XAie_TileLoc(0, 0), XAie_LockInit(2, 0))
XAie_LockSetValue(DevInst, XAie_TileLoc(0, 0), XAie_LockInit(3, 0))
XAie_LockSetValue(DevInst, XAie_TileLoc(0, 2), XAie_LockInit(0, 2))
XAie_LockSetValue(DevInst, XAie_TileLoc(0, 2), XAie_LockInit(1, 0))
XAie_LockSetValue(DevInst, XAie_TileLoc(0, 0), XAie_LockInit(0, 0))
XAie_LockSetValue(DevInst, XAie_TileLoc(0, 0), XAie_LockInit(1, 0))

dma_tile02_bd0 = XAie_DmaDesc()
XAie_DmaDescInit(DevInst, (dma_tile02_bd0), XAie_TileLoc(0, 2))
XAie_DmaSetLock((dma_tile02_bd0), XAie_LockInit(0, -1), XAie_LockInit(1, 1))
XAie_DmaSetAddrLen((dma_tile02_bd0), 0x400, 1024 * 4)
XAie_DmaSetNextBd((dma_tile02_bd0), 1, 1)
XAie_DmaEnableBd((dma_tile02_bd0))
XAie_DmaWriteBd(DevInst, (dma_tile02_bd0), XAie_TileLoc(0, 2), 0)

dma_tile02_bd1 = XAie_DmaDesc()
XAie_DmaDescInit(DevInst, (dma_tile02_bd1), XAie_TileLoc(0, 2))
XAie_DmaSetLock((dma_tile02_bd1), XAie_LockInit(0, -1), XAie_LockInit(1, 1))
XAie_DmaSetAddrLen((dma_tile02_bd1), 0x1400, 1024 * 4)
XAie_DmaSetNextBd((dma_tile02_bd1), 0, 1)
XAie_DmaEnableBd((dma_tile02_bd1))
XAie_DmaWriteBd(DevInst, (dma_tile02_bd1), XAie_TileLoc(0, 2), 1)

dma_tile02_bd2 = XAie_DmaDesc()
XAie_DmaDescInit(DevInst, (dma_tile02_bd2), XAie_TileLoc(0, 2))
XAie_DmaSetLock((dma_tile02_bd2), XAie_LockInit(1, -1), XAie_LockInit(0, 1))
XAie_DmaSetAddrLen((dma_tile02_bd2), 0x400, 1024 * 4)
XAie_DmaSetNextBd((dma_tile02_bd2), 3, 1)
XAie_DmaEnableBd((dma_tile02_bd2))
XAie_DmaWriteBd(DevInst, (dma_tile02_bd2), XAie_TileLoc(0, 2), 2)

dma_tile02_bd3 = XAie_DmaDesc()
XAie_DmaDescInit(DevInst, (dma_tile02_bd3), XAie_TileLoc(0, 2))
XAie_DmaSetLock((dma_tile02_bd3), XAie_LockInit(1, -1), XAie_LockInit(0, 1))
XAie_DmaSetAddrLen((dma_tile02_bd3), 0x1400, 1024 * 4)
XAie_DmaSetNextBd((dma_tile02_bd3), 2, 1)
XAie_DmaEnableBd((dma_tile02_bd3))
XAie_DmaWriteBd(DevInst, (dma_tile02_bd3), XAie_TileLoc(0, 2), 3)

XAie_DmaChannelPushBdToQueue(DevInst, XAie_TileLoc(0, 2), 0, DMA_S2MM, 0)
XAie_DmaChannelEnable(DevInst, XAie_TileLoc(0, 2), 0, DMA_S2MM)
XAie_DmaChannelPushBdToQueue(DevInst, XAie_TileLoc(0, 2), 0, DMA_MM2S, 2)
XAie_DmaChannelEnable(DevInst, XAie_TileLoc(0, 2), 0, DMA_MM2S)

# Core Stream Switch column 0 row 0
x = 0
y = 0
XAie_StrmConnCctEnable(DevInst, XAie_TileLoc(x, y), CTRL, 0, SOUTH, 0)
# configure DMA_<S2MM/MM2S>_<N>_Ctrl register
DmaChannelDescInst = XAie_DmaChannelDesc()
XAie_DmaChannelDescInit(DevInst, DmaChannelDescInst, XAie_TileLoc(x, y))
XAie_DmaChannelSetControllerId(DmaChannelDescInst, 0)
XAie_DmaWriteChannel(DevInst, DmaChannelDescInst, XAie_TileLoc(x, y), 0, DMA_S2MM)

# configure DMA_<S2MM/MM2S>_<N>_Ctrl register
DmaChannelDescInst = XAie_DmaChannelDesc()
XAie_DmaChannelDescInit(DevInst, DmaChannelDescInst, XAie_TileLoc(x, y))
XAie_DmaChannelSetControllerId(DmaChannelDescInst, 0)
XAie_DmaWriteChannel(DevInst, DmaChannelDescInst, XAie_TileLoc(x, y), 1, DMA_S2MM)

XAie_AieToPlIntfEnable(DevInst, XAie_TileLoc(x, y), 0, PLIF_WIDTH_32)
XAie_StrmConnCctEnable(DevInst, XAie_TileLoc(x, y), SOUTH, 3, NORTH, 0)
XAie_StrmConnCctEnable(DevInst, XAie_TileLoc(x, y), NORTH, 0, SOUTH, 2)
# Core Stream Switch column 0 row 2
x = 0
y = 2
XAie_StrmConnCctEnable(DevInst, XAie_TileLoc(x, y), SOUTH, 0, DMA, 0)
XAie_StrmConnCctEnable(DevInst, XAie_TileLoc(x, y), DMA, 0, SOUTH, 0)
# Core Stream Switch column 0 row 1
x = 0
y = 1
XAie_StrmConnCctEnable(DevInst, XAie_TileLoc(x, y), SOUTH, 0, NORTH, 0)
XAie_StrmConnCctEnable(DevInst, XAie_TileLoc(x, y), NORTH, 0, SOUTH, 0)
# ShimMux column 0 row 0
# NOTE ShimMux always connects from the south as directions are defined
# relative to the tile stream switch
x = 0
y = 0
XAie_EnableShimDmaToAieStrmPort(DevInst, XAie_TileLoc(x, y), 3)
XAie_EnableAieToShimDmaStrmPort(DevInst, XAie_TileLoc(x, y), 2)
configureHeader()
endCurrentCDOFileStream()

print("START: Core Enable Configuration")
startCDOFileStream(String(b"aie_cdo_enable.bin"))
FileHeader()
XAie_CoreEnable(DevInst, XAie_TileLoc(0, 2))
configureHeader()
endCurrentCDOFileStream()
