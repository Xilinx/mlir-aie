// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "Section.h"
#include "hrx_util.h"

#include <memory>
#include <sstream>

namespace {
class SectionKeyValueMetadata : public Section {
  void marshalToJSON(char *data, unsigned int size,
                     hrx::property_tree::ptree &tree) const override {
    hrx::property_tree::ptree metadata;
    if (!size) {
      metadata.add_child("key_values", hrx::property_tree::ptree{});
    } else {
      std::istringstream input(std::string(data, size));
      hrx::property_tree::read_json(input, metadata);
    }
    tree.add_child("keyvalue_metadata", metadata);
  }

  void marshalFromJSON(const hrx::property_tree::ptree &tree,
                       std::ostringstream &output) const override {
    hrx::property_tree::write_json(output, tree.get_child("keyvalue_metadata"),
                                   false);
  }

  static struct Init {
    Init() {
      auto info = std::make_unique<SectionInfo>(
          KEYVALUE_METADATA, "KEYVALUE_METADATA",
          hrx::factory<SectionKeyValueMetadata *>());
      info->nodeName = "keyvalue_metadata";
      info->supportedAddFormats = {FormatType::json};
      info->supportedDumpFormats = {FormatType::json};
      addSectionType(std::move(info));
    }
  } initializer;
};

SectionKeyValueMetadata::Init SectionKeyValueMetadata::initializer;
} // namespace
