
#include "XclBinSignature.h"
void signXclBinImage(const std::string& _fileOnDisk,
                     const std::string& _sPrivateKey,
                     const std::string& _sCertificate,
                     const std::string& _sDigestAlgorithm,
                     bool _bEnableDebugOutput) {}
void verifyXclBinImage(const std::string& _fileOnDisk,
                       const std::string& _sCertificate,
                       bool _bEnableDebugOutput) {}
void dumpSignatureFile(const std::string& _fileOnDisk,
                       const std::string& _signatureFile) {}
void getXclBinPKCSStats(const std::string& _xclBinFile,
                        XclBinPKCSImageStats& _xclBinPKCSImageStats) {}