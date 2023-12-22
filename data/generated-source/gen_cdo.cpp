#include "gen_cdo.h"
#include <cstring>
#include <iostream>
// We really should not do this!
#include "aie_control.cpp"
extern "C" {
#include "cdo_driver.h"
}

void initializeCDOGenerator(bool AXIdebug, bool endianness) {
  if (AXIdebug)
    EnAXIdebug(); // Enables AXI-MM prints for configs being added in CDO,
                  // helpful for debugging
  setEndianness(endianness);
}

void addInitConfigToCDO(const std::string &workDirPath) {
  ppgraph_init(workDirPath);
}

void addCoreEnableToCDO() { ppgraph_core_enable(); }

void addErrorHandlingToCDO() { enableErrorHandling(); }

void addAieElfsToCDO(const std::string &workDirPath) {
  std::vector<std::string> elfInfoPath;
  if (!ppgraph_load_elf(workDirPath, elfInfoPath))
    exit(EXIT_FAILURE);
}

void generateCDOBinariesSeparately(const std::string &workDirPath,
                                   bool AXIdebug) {

  // aie_cdo_error_handling.bin
  const std::string errorHandlingCDOFilePath =
      workDirPath + "aie_cdo_error_handling.bin";
  if (AXIdebug)
    std::cout << "START: Error Handling Configuration\n";
  startCDOFileStream(errorHandlingCDOFilePath.c_str());
  FileHeader();
  addErrorHandlingToCDO();
  configureHeader();
  endCurrentCDOFileStream();
  if (AXIdebug)
    std::cout << "DONE: Error Handling Configuration\n\n";

  // aie_cdo_elfs.bin
  const std::string elfsCDOFilePath = workDirPath + "aie_cdo_elfs.bin";
  if (AXIdebug)
    std::cout << "START: AIE ELF Configuration\n";
  startCDOFileStream(elfsCDOFilePath.c_str());
  FileHeader();
  addAieElfsToCDO(workDirPath);
  configureHeader();
  endCurrentCDOFileStream();
  if (AXIdebug)
    std::cout << "DONE: AIE ELF Configuration\n\n";

  // aie_cdo_init.bin
  const std::string initCfgCDOFilePath = workDirPath + "aie_cdo_init.bin";
  if (AXIdebug)
    std::cout << "START: Initial Configuration (SHIM and AIE Array)\n";
  startCDOFileStream(initCfgCDOFilePath.c_str());
  FileHeader();
  addInitConfigToCDO(workDirPath);
  configureHeader();
  endCurrentCDOFileStream();
  if (AXIdebug)
    std::cout << "DONE: Initial Configuration (SHIM and AIE Array)\n\n";

  // aie_cdo_enable.bin
  const std::string coreEnableCDOFilePath = workDirPath + "aie_cdo_enable.bin";
  if (AXIdebug)
    std::cout << "START: Core Enable Configuration\n";
  startCDOFileStream(coreEnableCDOFilePath.c_str());
  FileHeader();
  addCoreEnableToCDO();
  configureHeader();
  endCurrentCDOFileStream();
  if (AXIdebug)
    std::cout << "DONE: Core Enable Configuration\n\n";
}
