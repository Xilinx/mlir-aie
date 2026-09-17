// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "StackSizeAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace xilinx::aiecc;

int main(int argc, char **argv) {
  assert(argc == 13);
  using Kind = LutOperand::Kind;
  auto textPairs = readLutPairsFromIR(argv[1]);
  auto bitcodePairs = readLutPairsFromIR(argv[3]);
  assert(textPairs && textPairs->size() == 12);
  assert(bitcodePairs && bitcodePairs->size() == 12);
  assert(!readLutPairsFromIR(argv[7]));
  for (int i = 1; i <= 3; ++i) {
    auto pairs = readLutPairsFromObject(argv[i]);
    assert(pairs && pairs->size() == 12);
    assert((*pairs)[0].function == "gather_with_unrelated_blend");
    assert((*pairs)[0].a.symbol == "table_a");
    assert((*pairs)[0].b.symbol == "table_b");
    assert((*pairs)[1].function == "unresolved_gather");
    assert((*pairs)[1].a.kind == Kind::Unknown);
    assert((*pairs)[1].b.kind == Kind::Unknown);
    assert((*pairs)[2].function == "offset_gather");
    assert((*pairs)[2].a.kind == Kind::Unknown);
    assert((*pairs)[2].b.symbol == "table_b");
    assert((*pairs)[3].function == "same_table");
    assert((*pairs)[3].a.symbol == "table_a");
    assert((*pairs)[3].b.symbol == "table_a");
    assert((*pairs)[4].function == "stack_table");
    assert((*pairs)[4].a.kind == Kind::Stack);
    assert((*pairs)[4].b.kind == Kind::Param);
    assert((*pairs)[4].b.paramIndex == 0);
    assert((*pairs)[5].function == "aie2_gather");
    assert((*pairs)[5].a.symbol == "table_a");
    assert((*pairs)[5].b.symbol == "table_b");
    assert((*pairs)[6].function == "partially_resolved_select");
    assert((*pairs)[6].a.kind == Kind::Unknown);
    assert((*pairs)[6].b.kind == Kind::Unknown);
    for (int j = 7; j < 11; ++j) {
      assert((*pairs)[j].function ==
             (j < 9 ? "aie2_gather32" : "aie2p_gather64"));
      assert((*pairs)[j].a.symbol == "table_a");
      assert((*pairs)[j].b.symbol == "table_b");
    }
    assert((*pairs)[11].function == "unresolved_vsel_gather");
    assert((*pairs)[11].a.kind == Kind::Unknown);
    assert((*pairs)[11].b.kind == Kind::Unknown);
  }
  assert(!readLutPairsFromObject(argv[4]));
  assert(!readLutPairsFromObject(argv[5]));
  assert(!readLutPairsFromObject(argv[6]));
  assert(!readLutPairsFromObject(argv[7]));
  auto assertions = readBankAssertionsFromObjects({std::string(argv[4])});
  assert(assertions.size() == 6);
  for (const auto &assertion : assertions) {
    if (assertion.symbol == "bank_ab") {
      assert(assertion.banks.size() == 2);
      assert(assertion.banks[0] == 0 && assertion.banks[1] == 1);
    } else {
      assert(assertion.banks.size() == 1);
      assert(assertion.banks[0] == assertion.symbol[4] - '0');
    }
  }
  llvm::StringMap<uint64_t> sizes;
  auto addresses = readDataSymbolAddresses(argv[4], 0, &sizes);
  assert(!addresses.contains("undefined"));
  assert(sizes.lookup("bank0") == 4 * sizeof(int));
  auto duplicates = readDataSymbolAddresses(argv[10], 0);
  assert(!duplicates.contains("table"));
  int64_t base = duplicates.lookup("bank_analysis_anchor");
  assert(base > 0);
  auto duplicateAssertions = readBankAssertionsFromObjects(
      {std::string(argv[8]), std::string(argv[9])});
  int tableAssertions = 0;
  for (const auto &assertion : duplicateAssertions)
    tableAssertions += assertion.symbol == "table";
  assert(tableAssertions == 0);
  assert(checkBankPlacements(argv[10], duplicateAssertions, base, 4096, 4)
             .empty());
  BankAssertion contradiction{"bank_analysis_anchor", ".aie.bank1", {1}};
  auto violations =
      checkBankPlacements(argv[10], {contradiction}, base, 4096, 4);
  assert(violations.size() == 1 && violations[0].actualBank == 0);
  auto collected = readDataSymbolAddresses(argv[11], 0);
  assert(collected.contains("table"));
  int64_t collectedBase = collected.lookup("bank_analysis_anchor");
  assert(collectedBase > 0);
  assert(
      checkBankPlacements(argv[11], duplicateAssertions, collectedBase, 4096, 4)
          .empty());
  auto unpinnedAssertions = readBankAssertionsFromObjects(
      {std::string(argv[8]), std::string(argv[12])});
  for (const auto &assertion : unpinnedAssertions)
    assert(assertion.symbol != "table");
  auto singleAssertions = readBankAssertionsFromObjects(
      {std::string(argv[8]), std::string(argv[8])});
  tableAssertions = 0;
  for (const auto &assertion : singleAssertions)
    tableAssertions += assertion.symbol == "table";
  assert(tableAssertions == 1);
  llvm::outs() << "LUT analysis: all checks passed\n";
}
