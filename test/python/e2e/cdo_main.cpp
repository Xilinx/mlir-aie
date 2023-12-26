#include "cdo_driver.h"

extern "C" {
#include <xaiengine/xaie_core.h>
#include <xaiengine/xaie_elfloader.h>
#include <xaiengine/xaie_interrupt.h>
#include <xaiengine/xaie_plif.h>
#include <xaiengine/xaie_ss.h>
}

#include <getopt.h>
#include <iostream>
#include <unistd.h>
#include <vector>

/************************** Constants/Macros *****************************/
#define HW_GEN XAIE_DEV_GEN_AIEML
#define XAIE_NUM_ROWS 6
#define XAIE_NUM_COLS 5
#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_MEM_TILE_NUM_ROWS 1
#define XAIE_AIE_TILE_ROW_START 2
#define XAIE_AIE_TILE_NUM_ROWS 4
#define FOR_WRITE 0
#define FOR_READ 1
#define XAIE_PARTITION_BASE_ADDR 0x0

/***************************** Includes *********************************/

#define __mlir_aie_try(x) x
static XAie_DmaDimDesc *mlirAieAllocDimDesc(size_t ndims) {
  XAie_DmaDimDesc *ret = NULL;
  ret = (XAie_DmaDimDesc *)calloc(sizeof(XAie_DmaDimDesc), ndims);
  if (NULL == ret) {
    fprintf(stderr, "Allocating DmaDimDesc failed.\n");
  }
  return ret;
}

XAie_InstDeclare(DevInst, &ConfigPtr); // Declare global device instance

bool ppgraphLoadElf(const std::string &workPath,
                    std::vector<std::string> &elfInfoPath) {
  std::string workDir = (workPath.empty() ? "Work" : workPath);
  {
    if (XAie_LoadElf(&DevInst, XAie_TileLoc(0, 2),
                     (workDir + "/core_0_2.elf").c_str(),
                     XAIE_ENABLE) != XAIE_OK) {
      std::cerr << "ERROR: Failed to load elf for core at " << workDir
                << std::endl;
      return false;
    }
  }
  return true;
} // ppgraph_load_elf

void ppgraphCoreEnable() {
  XAie_CoreEnable(&DevInst, XAie_TileLoc(0, 2));
} // ppgraph_core_enable

void enableErrorHandling() {
  XAie_ErrorHandlingInit(&DevInst);
} // enableErrorHandling

void ppgraphInit(const std::string &workPath) {
  XAie_CoreReset(&DevInst, XAie_TileLoc(0, 2));
  XAie_CoreUnreset(&DevInst, XAie_TileLoc(0, 2));
  for (int l = 0; l < 16; l++)
    XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 2), XAie_LockInit(l, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 0), XAie_LockInit(0, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 0), XAie_LockInit(1, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 0), XAie_LockInit(2, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 0), XAie_LockInit(3, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 1), XAie_LockInit(0, 2));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 1), XAie_LockInit(1, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 1), XAie_LockInit(2, 2));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 1), XAie_LockInit(3, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 2), XAie_LockInit(0, 2));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 2), XAie_LockInit(1, 0));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 2), XAie_LockInit(2, 2));
  XAie_LockSetValue(&DevInst, XAie_TileLoc(0, 2), XAie_LockInit(3, 0));
  XAie_DmaDesc dmaTile02Bd0;
  XAie_DmaDescInit(&DevInst, &(dmaTile02Bd0), XAie_TileLoc(0, 2));
  XAie_DmaSetLock(&(dmaTile02Bd0), XAie_LockInit(0, -1), XAie_LockInit(1, 1));
  XAie_DmaSetAddrLen(&(dmaTile02Bd0), /* addrA */ 0x400, /* len */ 8 * 4);
  XAie_DmaSetNextBd(&(dmaTile02Bd0), /* nextbd */ 1, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile02Bd0));
  XAie_DmaWriteBd(&DevInst, &(dmaTile02Bd0), XAie_TileLoc(0, 2), /* bd */ 0);
  XAie_DmaDesc dmaTile02Bd1;
  XAie_DmaDescInit(&DevInst, &(dmaTile02Bd1), XAie_TileLoc(0, 2));
  XAie_DmaSetLock(&(dmaTile02Bd1), XAie_LockInit(0, -1), XAie_LockInit(1, 1));
  XAie_DmaSetAddrLen(&(dmaTile02Bd1), /* addrA */ 0x420, /* len */ 8 * 4);
  XAie_DmaSetNextBd(&(dmaTile02Bd1), /* nextbd */ 0, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile02Bd1));
  XAie_DmaWriteBd(&DevInst, &(dmaTile02Bd1), XAie_TileLoc(0, 2), /* bd */ 1);
  XAie_DmaDesc dmaTile02Bd2;
  XAie_DmaDescInit(&DevInst, &(dmaTile02Bd2), XAie_TileLoc(0, 2));
  XAie_DmaSetLock(&(dmaTile02Bd2), XAie_LockInit(3, -1), XAie_LockInit(2, 1));
  XAie_DmaSetAddrLen(&(dmaTile02Bd2), /* addrA */ 0x440, /* len */ 8 * 4);
  XAie_DmaSetNextBd(&(dmaTile02Bd2), /* nextbd */ 3, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile02Bd2));
  XAie_DmaWriteBd(&DevInst, &(dmaTile02Bd2), XAie_TileLoc(0, 2), /* bd */ 2);
  XAie_DmaDesc dmaTile02Bd3;
  XAie_DmaDescInit(&DevInst, &(dmaTile02Bd3), XAie_TileLoc(0, 2));
  XAie_DmaSetLock(&(dmaTile02Bd3), XAie_LockInit(3, -1), XAie_LockInit(2, 1));
  XAie_DmaSetAddrLen(&(dmaTile02Bd3), /* addrA */ 0x460, /* len */ 8 * 4);
  XAie_DmaSetNextBd(&(dmaTile02Bd3), /* nextbd */ 2, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile02Bd3));
  XAie_DmaWriteBd(&DevInst, &(dmaTile02Bd3), XAie_TileLoc(0, 2), /* bd */ 3);
  XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0, 2), /* ChNum */ 0,
                               /* dmaDir */ DMA_S2MM, /* BdNum */ 0);
  XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0, 2), /* ChNum */ 0,
                        /* dmaDir */ DMA_S2MM);
  XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0, 2), /* ChNum */ 0,
                               /* dmaDir */ DMA_MM2S, /* BdNum */ 2);
  XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0, 2), /* ChNum */ 0,
                        /* dmaDir */ DMA_MM2S);
  XAie_DmaDesc dmaTile01Bd0;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd0), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd0), XAie_LockInit(64, -1), XAie_LockInit(65, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd0), /* addrA */ 0x80000, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd0), /* nextbd */ 1, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd0));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd0), XAie_TileLoc(0, 1), /* bd */ 0);
  XAie_DmaDesc dmaTile01Bd1;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd1), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd1), XAie_LockInit(64, -1), XAie_LockInit(65, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd1), /* addrA */ 0x80040, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd1), /* nextbd */ 0, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd1));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd1), XAie_TileLoc(0, 1), /* bd */ 1);
  XAie_DmaDesc dmaTile01Bd2;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd2), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd2), XAie_LockInit(65, -1), XAie_LockInit(64, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd2), /* addrA */ 0x80000, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd2), /* nextbd */ 3, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd2));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd2), XAie_TileLoc(0, 1), /* bd */ 2);
  XAie_DmaDesc dmaTile01Bd3;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd3), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd3), XAie_LockInit(65, -1), XAie_LockInit(64, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd3), /* addrA */ 0x80040, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd3), /* nextbd */ 2, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd3));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd3), XAie_TileLoc(0, 1), /* bd */ 3);
  XAie_DmaDesc dmaTile01Bd24;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd24), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd24), XAie_LockInit(67, -1),
                  XAie_LockInit(66, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd24), /* addrA */ 0x80080, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd24), /* nextbd */ 25, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd24));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd24), XAie_TileLoc(0, 1),
                  /* bd */ 24);
  XAie_DmaDesc dmaTile01Bd25;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd25), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd25), XAie_LockInit(67, -1),
                  XAie_LockInit(66, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd25), /* addrA */ 0x800C0, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd25), /* nextbd */ 24, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd25));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd25), XAie_TileLoc(0, 1),
                  /* bd */ 25);
  XAie_DmaDesc dmaTile01Bd26;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd26), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd26), XAie_LockInit(66, -1),
                  XAie_LockInit(67, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd26), /* addrA */ 0x80080, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd26), /* nextbd */ 27, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd26));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd26), XAie_TileLoc(0, 1),
                  /* bd */ 26);
  XAie_DmaDesc dmaTile01Bd27;
  XAie_DmaDescInit(&DevInst, &(dmaTile01Bd27), XAie_TileLoc(0, 1));
  XAie_DmaSetLock(&(dmaTile01Bd27), XAie_LockInit(66, -1),
                  XAie_LockInit(67, 1));
  XAie_DmaSetAddrLen(&(dmaTile01Bd27), /* addrA */ 0x800C0, /* len */ 16 * 4);
  XAie_DmaSetNextBd(&(dmaTile01Bd27), /* nextbd */ 26, /* enableNextBd */ 1);
  XAie_DmaEnableBd(&(dmaTile01Bd27));
  XAie_DmaWriteBd(&DevInst, &(dmaTile01Bd27), XAie_TileLoc(0, 1),
                  /* bd */ 27);
  XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 0,
                               /* dmaDir */ DMA_S2MM, /* BdNum */ 0);
  XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 0,
                        /* dmaDir */ DMA_S2MM);
  XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 0,
                               /* dmaDir */ DMA_MM2S, /* BdNum */ 2);
  XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 0,
                        /* dmaDir */ DMA_MM2S);
  XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 1,
                               /* dmaDir */ DMA_MM2S, /* BdNum */ 24);
  XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 1,
                        /* dmaDir */ DMA_MM2S);
  XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 1,
                               /* dmaDir */ DMA_S2MM, /* BdNum */ 26);
  XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0, 1), /* ChNum */ 1,
                        /* dmaDir */ DMA_S2MM);
  int x, y;
  // Core Stream Switch column 0 row 0
  x = 0;
  y = 0;
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), CTRL, 0, SOUTH, 0);
  {
    // configure DMA_<S2MM/MM2S>_<N>_Ctrl register
    XAie_DmaChannelDesc dmaChannelDescInst;
    XAie_DmaChannelDescInit(&DevInst, &dmaChannelDescInst, XAie_TileLoc(x, y));
    XAie_DmaChannelSetControllerId(&dmaChannelDescInst, 0);
    XAie_DmaWriteChannel(&DevInst, &dmaChannelDescInst, XAie_TileLoc(x, y), 0,
                         DMA_S2MM);
  }

  {
    // configure DMA_<S2MM/MM2S>_<N>_Ctrl register
    XAie_DmaChannelDesc dmaChannelDescInst;
    XAie_DmaChannelDescInit(&DevInst, &dmaChannelDescInst, XAie_TileLoc(x, y));
    XAie_DmaChannelSetControllerId(&dmaChannelDescInst, 0);
    XAie_DmaWriteChannel(&DevInst, &dmaChannelDescInst, XAie_TileLoc(x, y), 1,
                         DMA_S2MM);
  }

  XAie_AieToPlIntfEnable(&DevInst, XAie_TileLoc(x, y), 0, PLIF_WIDTH_32);
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), SOUTH, 3, NORTH, 0);
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), NORTH, 0, SOUTH, 2);
  // Core Stream Switch column 0 row 1
  x = 0;
  y = 1;
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), SOUTH, 0, DMA, 0);
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), DMA, 0, NORTH, 0);
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), DMA, 1, SOUTH, 0);
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), NORTH, 0, DMA, 1);
  // Core Stream Switch column 0 row 2
  x = 0;
  y = 2;
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), SOUTH, 0, DMA, 0);
  XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x, y), DMA, 0, SOUTH, 0);
  // ShimMux column 0 row 0
  // NOTE ShimMux always connects from the south as directions are defined
  // relative to the tile stream switch
  x = 0;
  y = 0;
  XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(x, y), 3);
  XAie_EnableAieToShimDmaStrmPort(&DevInst, XAie_TileLoc(x, y), 2);
} // ppgraph_init

class InitializeAIEControl {
public:
  InitializeAIEControl() {
    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT,
                     XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_MEM_TILE_ROW_START,
                     XAIE_MEM_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    XAie_SetupPartitionConfig(&DevInst, XAIE_PARTITION_BASE_ADDR, 1, 1);

    XAie_CfgInitialize(&(DevInst), &ConfigPtr);

    XAie_SetIOBackend(
        &(DevInst),
        XAIE_IO_BACKEND_CDO); // Set aiengine driver library to run for CDO Mode
    XAie_UpdateNpiAddr(&(DevInst), 0x0);
  }
} initAIEControl;

void initializeCDOGenerator(bool axIdebug, bool endianness) {
  if (axIdebug)
    EnAXIdebug(); // Enables AXI-MM prints for configs being added in CDO,
                  // helpful for debugging
  setEndianness(endianness);
}

void addInitConfigToCDO(const std::string &workDirPath) {
  ppgraphInit(workDirPath);
}

void addCoreEnableToCDO() { ppgraphCoreEnable(); }

void addErrorHandlingToCDO() { enableErrorHandling(); }

void addAieElfsToCDO(const std::string &workDirPath) {
  std::vector<std::string> elfInfoPath;
  if (!ppgraphLoadElf(workDirPath, elfInfoPath))
    exit(EXIT_FAILURE);
}

void generateCDOBinariesSeparately(const std::string &workDirPath,
                                   bool axIdebug) {

  // aie_cdo_error_handling.bin
  const std::string errorHandlingCDOFilePath =
      workDirPath + "/aie_cdo_error_handling.bin";
  if (axIdebug)
    std::cout << "START: Error Handling Configuration\n";
  startCDOFileStream(errorHandlingCDOFilePath.c_str());
  FileHeader();
  addErrorHandlingToCDO();
  configureHeader();
  endCurrentCDOFileStream();
  if (axIdebug)
    std::cout << "DONE: Error Handling Configuration\n\n";

  // aie_cdo_elfs.bin
  const std::string elfsCDOFilePath = workDirPath + "/aie_cdo_elfs.bin";
  if (axIdebug)
    std::cout << "START: AIE ELF Configuration\n";
  startCDOFileStream(elfsCDOFilePath.c_str());
  FileHeader();
  addAieElfsToCDO(workDirPath);
  configureHeader();
  endCurrentCDOFileStream();
  if (axIdebug)
    std::cout << "DONE: AIE ELF Configuration\n\n";

  // aie_cdo_init.bin
  const std::string initCfgCDOFilePath = workDirPath + "/aie_cdo_init.bin";
  if (axIdebug)
    std::cout << "START: Initial Configuration (SHIM and AIE Array)\n";
  startCDOFileStream(initCfgCDOFilePath.c_str());
  FileHeader();
  addInitConfigToCDO(workDirPath);
  configureHeader();
  endCurrentCDOFileStream();
  if (axIdebug)
    std::cout << "DONE: Initial Configuration (SHIM and AIE Array)\n\n";

  // aie_cdo_enable.bin
  const std::string coreEnableCDOFilePath = workDirPath + "/aie_cdo_enable.bin";
  if (axIdebug)
    std::cout << "START: Core Enable Configuration\n";
  startCDOFileStream(coreEnableCDOFilePath.c_str());
  FileHeader();
  addCoreEnableToCDO();
  configureHeader();
  endCurrentCDOFileStream();
  if (axIdebug)
    std::cout << "DONE: Core Enable Configuration\n\n";
}

int main(int argc, char **argv) {
  std::string workDirPath;
  bool axIdebug = false;
  bool endianness = byte_ordering::Little_Endian;
  int opt;
  static struct option longOptions[] = {
      /* name, has_arg, flag, val */
      {"help", no_argument, nullptr, 0},
      {"aximm-dump", no_argument, nullptr, 1},
      {"big-endian-cdo", no_argument, nullptr, 2},
      {"work-dir-path", required_argument, nullptr, 3}};
  while (true) {
    int optIndex = 0;
    opt = getopt_long(argc, argv, "habw:", longOptions, &optIndex);
    if (opt == -1)
      break;
    switch (opt) {
    case 0:
    case 'h':
      std::cout << "\n*********************************************************"
                   "***********************************************************"
                   "*********** \n\n\t";
      std::cout
          << "\n This script is used in AI Engine Compiler to generate CDO "
             "binaries and record statically allocated resources. Files "
             "generated by this script are:\n\n"
             "\t\t\t >> aie_cdo_init.bin            : Initial Configurations "
             "CDO. Stream Switch, DMA, SHIM etc...\n"
             "\t\t\t >> aie_cdo_enable.bin          : Core Enable CDO\n"
             "\t\t\t >> aie_cdo_debug.bin           : Core Debug Halt CDO \n"
             "\t\t\t >> aie_cdo_reset.bin           : Partition Reset CDO \n"
             "\t\t\t >> aie_cdo_mem_clear.bin       : Partition DM/PM clearing "
             "CDO\n"
             "\t\t\t >> aie_cdo_error_handling.bin  : Error Handling CDO \n"
             "\t\t\t >> aie_cdo_clock_gating.bin    : Clock Gating CDO \n"
             "\t\t\t >> aie_cdo_elfs.bin            : AIE ELF packed into CDO "
             "(Default ECC-scrubbing enabled)\n"
             "\t\t\t >> aie_resources.bin           : Statically allocated "
             "resource file";
      std::cout << "\n*********************************************************"
                   "***********************************************************"
                   "*********** \n\n\t";
      std::cout << "\nAccepted options are :\n\n\t"
                   "1) --help or -h: Displays list of options accepted by cdo "
                   "executable.(Usage: --help) \n\n\t"
                   "2) --aximm-dump or -a: Generates AXI-MM Dump helpful for "
                   "debugging.(Usage: --aximm-dump)\n\n\t"
                   "3) --big-endian-cdo or -b: Generates AIE CDO in big endian "
                   "format, default is little-endian (Usage: --big-endian-cdo) "
                   "\n\n  ";
      return EXIT_SUCCESS;
    case 1:
    case 'a':
      axIdebug = true;
      break;
    case 2:
    case 'b':
      endianness = byte_ordering::Big_Endian;
      break;
    case 3:
    case 'w':
      workDirPath = optarg;
      break;
    case '?':
      return EXIT_FAILURE;
      break;
    default:
      std::cout
          << "getopt returned char code which is not handled, returned code:"
          << opt << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (optind < argc) {
    std::cout << "Value provided without any option(non-optioned argv): ";
    while (optind < argc)
      std::cout << argv[optind++];
    std::cout << "\n";
    return EXIT_FAILURE;
  }

  initializeCDOGenerator(axIdebug, endianness);
  generateCDOBinariesSeparately(workDirPath, axIdebug);
  return EXIT_SUCCESS;
}
