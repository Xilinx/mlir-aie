// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/* cdo_driver.h  */
#ifndef _CDO_DRIVER_H_
#define _CDO_DRIVER_H_

#include <cstdint> // uint

extern "C" {
enum byte_ordering { Little_Endian, Big_Endian };
void startCDOFileStream(const char *cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void SectionHeader();
void configureHeader();
unsigned int getPadBytesForDmaWrCmdAlignment(uint32_t DmaCmdLength);
void insertNoOpCommand(unsigned int numPadBytes);
void insertDmaWriteCmdHdr(uint32_t DmaCmdLength);
void disableDmaCmdAlignment();
};

#endif /* _CDO_DRIVER_H_ */
