#include "generated_txn.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::vector<uint32_t> loadWords(const char *path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    throw std::runtime_error(std::string("failed to open: ") + path);
  auto sz = f.tellg();
  if (sz < 0)
    throw std::runtime_error(std::string("failed to determine size of: ") +
                             path);
  f.seekg(0, std::ios::beg);
  std::vector<uint32_t> data(static_cast<size_t>(sz) / sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(data.data()), sz);
  if (f.gcount() != sz)
    throw std::runtime_error(std::string("short read from: ") + path);
  return data;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <static-inst-bin>\n";
    return 2;
  }

  constexpr uint32_t sizeElems = IN1_SIZE;

  auto dynamicTxn = generate_txn_sequence(sizeElems);
  auto staticTxn = loadWords(argv[1]);

  if (dynamicTxn == staticTxn) {
    std::cout << "dynamic and static TXN streams match (" << dynamicTxn.size()
              << " words)\n";
    return 0;
  }

  std::cerr << "dynamic and static TXN streams differ: dynamic="
            << dynamicTxn.size() << " static=" << staticTxn.size() << "\n";
  size_t limit = std::min(dynamicTxn.size(), staticTxn.size());
  for (size_t i = 0; i < limit; ++i) {
    if (dynamicTxn[i] != staticTxn[i]) {
      std::cerr << "first diff at word " << i << ": dynamic=" << dynamicTxn[i]
                << " static=" << staticTxn[i] << "\n";
      break;
    }
  }
  return 1;
}
