#ifndef _GEN_CDO_H_
#define _GEN_CDO_H_

#include <string>
#include <vector>

void initializeCDOGenerator(bool AXIdebug, bool endianness);

void generateCDOBinariesSeparately(const std::string &outputDir, bool AXIdebug);

#endif /* _GEN_CDO_H_ */
