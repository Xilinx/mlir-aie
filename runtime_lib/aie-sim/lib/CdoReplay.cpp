//===- CdoReplay.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The CDO container format, and the five commands mlir-aie emits into it.
//
// Provenance is bootgen's own writer, third_party/bootgen/cdo-driver: the
// header layout is `cdoHeader` plus FileHeader() (cdo_driver.c:41,158-166) and
// each command's word order is the fwrite sequence in the function that emits
// it. Command ids are cdo_driver.h:26-31. Nothing here is inferred from the
// bytes of a sample file -- a format read off examples is a format that breaks
// on the first design that uses a field the samples happened not to.
//
//===----------------------------------------------------------------------===//

#include "aiesim/CdoReplay.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

// cdo_driver.h:26-31.
constexpr uint32_t kCmdDmaWrite = 0x105;   // Block write; payload follows addr.
constexpr uint32_t kCmdMaskPoll64 = 0x106;
constexpr uint32_t kCmdMaskWrite64 = 0x107;
constexpr uint32_t kCmdWrite64 = 0x108;
constexpr uint32_t kCmdNoOperation = 0x111;

// cdo_driver.c:41. "CDO\0" little-endian.
constexpr uint32_t kIdentWord = 0x004F4443;

// A command header is `Length[23:16] | API-ID[15:0]`, where Length counts the
// PAYLOAD words that follow it (cdo_Write32's 3 = Addr1 + Addr0 + Data, and so
// on for each emitter). The field is 8 bits, so a block write longer than that
// escapes: insertDmaWriteCmdHdr (cdo_driver.c:246-259) writes 255 in the field
// and the real length as one extra word before the payload.
constexpr uint32_t kLongBlockWriteLength = 255;

struct Reader {
  const uint32_t *words;
  size_t count;
  size_t pos = 0;

  bool have(size_t n) const { return pos + n <= count; }
  uint32_t at(size_t i) const { return words[pos + i]; }
};

std::string atWord(size_t pos) {
  return " (at word " + std::to_string(pos) + ")";
}

} // namespace

bool aiesim::replayCdo(Array &array, const void *image, size_t sizeBytes,
                       uint64_t base, CdoReplayStats &stats, std::string &error,
                       uint64_t maskPollCycles) {
  if (sizeBytes % 4 != 0) {
    error = "CDO image is not a whole number of 32-bit words";
    return false;
  }
  // The whole format is 32-bit words and every emitter writes through
  // LEfwrite, so a byte-wise copy into a word vector is both the alignment fix
  // and the documented endianness (little, on any host this builds for).
  std::vector<uint32_t> words(sizeBytes / 4);
  std::memcpy(words.data(), image, sizeBytes);

  // FileHeader(): NumWords, IdentWord, Version, CDOLength, CheckSum.
  // `NumWords` is 4 and counts the four that follow it, so the header is 5.
  constexpr size_t kHeaderWords = 5;
  if (words.size() < kHeaderWords) {
    error = "CDO image is shorter than its own header";
    return false;
  }
  if (words[1] != kIdentWord) {
    char buf[96];
    std::snprintf(buf, sizeof(buf),
                  "not a CDO image: ident word is 0x%08X, expected 0x%08X",
                  words[1], kIdentWord);
    error = buf;
    return false;
  }
  // configureHeader(), cdo_driver.c:375. Cheap, and it catches a truncated or
  // concatenated file before the command loop reads a length out of garbage.
  uint32_t checksum = ~(words[0] + words[1] + words[2] + words[3]);
  if (checksum != words[4]) {
    char buf[112];
    std::snprintf(buf, sizeof(buf),
                  "CDO header checksum is 0x%08X, computed 0x%08X", words[4],
                  checksum);
    error = buf;
    return false;
  }
  if (words[3] != words.size() - kHeaderWords) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "CDO header declares %u command words but the file holds %zu",
                  words[3], words.size() - kHeaderWords);
    error = buf;
    return false;
  }

  Reader r{words.data(), words.size(), kHeaderWords};
  while (r.pos < r.count) {
    const size_t cmdPos = r.pos;
    const uint32_t header = r.at(0);
    const uint32_t cmd = header & 0xFFFFu;
    uint32_t length = (header >> 16) & 0xFFu;
    size_t payload = 1; // Words consumed before the payload proper.

    if (cmd == kCmdDmaWrite && length == kLongBlockWriteLength) {
      if (!r.have(2)) {
        error = "CDO block write claims a long length but the file ends" +
                atWord(cmdPos);
        return false;
      }
      length = r.at(1);
      payload = 2;
    }
    if (!r.have(payload + length)) {
      char buf[112];
      std::snprintf(buf, sizeof(buf),
                    "CDO command 0x%03X declares %u payload words but only %zu "
                    "remain",
                    cmd, length, r.count - r.pos - payload);
      error = buf + atWord(cmdPos);
      return false;
    }

    // Every address is emitted high word first (Addr1 then Addr0), which is
    // the one place this format is not little-endian-by-word.
    auto address = [&r, payload, base]() {
      return base | (static_cast<uint64_t>(r.at(payload)) << 32) |
             r.at(payload + 1);
    };

    switch (cmd) {
    case kCmdWrite64: {
      array.write32(address(), r.at(payload + 2));
      ++stats.write32;
      break;
    }
    case kCmdMaskWrite64: {
      uint64_t addr = address();
      uint32_t mask = r.at(payload + 2);
      uint32_t data = r.at(payload + 3);
      // Read-modify-write, exactly as XAie_SimIO_MaskWrite32 does it
      // (xaie_sim.c). The read is the reason a masked write can land on a
      // register nothing models and fault: that is the intended behaviour,
      // since a mask-write onto an unmodelled register would otherwise apply
      // the design's bits to a fabricated zero.
      array.write32(addr, (array.read32(addr) & ~mask) | (data & mask));
      ++stats.maskWrite32;
      break;
    }
    case kCmdDmaWrite: {
      // cdo_BlockWrite32: DmaCmdLength is size + 2, the 2 being the address.
      uint64_t addr = address();
      uint32_t n = length - 2;
      for (uint32_t i = 0; i < n; ++i)
        array.write32(addr + 4ull * i, r.at(payload + 2 + i));
      ++stats.blockWrite32;
      stats.blockWriteWords += n;
      break;
    }
    case kCmdMaskPoll64: {
      uint64_t addr = address();
      uint32_t mask = r.at(payload + 2);
      uint32_t expected = r.at(payload + 3);
      // The fifth payload word is a wall-clock timeout in milliseconds, which
      // has no meaning here; see kDefaultMaskPollCycles.
      uint64_t deadline = array.cycle() + maskPollCycles;
      // Array::read32 advances the clock by one cycle, so this loop IS the
      // thing that lets the polled-for state arrive -- the same mechanism that
      // makes an aie-rt host wait work against this model (Array.cpp's
      // kCyclesPerHostRead comment).
      while ((array.read32(addr) & mask) != expected &&
             array.cycle() < deadline) {
      }
      if ((array.read32(addr) & mask) != expected)
        ++stats.maskPollTimedOut;
      ++stats.maskPoll;
      break;
    }
    case kCmdNoOperation:
      // Padding, inserted to 16-byte-align the payload of the block write that
      // follows (insertNoOpCommand, cdo_driver.c:231-247).
      ++stats.noOp;
      break;
    default: {
      char buf[112];
      std::snprintf(buf, sizeof(buf),
                    "unknown CDO command 0x%03X; refusing to skip it, because "
                    "the commands after it would be read at the wrong offset",
                    cmd);
      error = buf + atWord(cmdPos);
      return false;
    }
    }

    r.pos += payload + length;
  }
  return true;
}

bool aiesim::replayCdoFile(Array &array, const std::string &path, uint64_t base,
                           CdoReplayStats &stats, std::string &error,
                           uint64_t maskPollCycles) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    error = "cannot open CDO file: " + path;
    return false;
  }
  std::vector<uint8_t> image;
  uint8_t buf[65536];
  while (size_t n = std::fread(buf, 1, sizeof(buf), f))
    image.insert(image.end(), buf, buf + n);
  bool readFailed = std::ferror(f) != 0;
  std::fclose(f);
  if (readFailed) {
    error = "error reading CDO file: " + path;
    return false;
  }
  if (!replayCdo(array, image.data(), image.size(), base, stats, error,
                 maskPollCycles)) {
    error += " [" + path + "]";
    return false;
  }
  return true;
}
