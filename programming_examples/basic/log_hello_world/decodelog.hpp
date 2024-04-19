//===- decodelog.hpp ---------------------------------------000---*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt/xrt_bo.h"
#include <boost/tokenizer.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <string>

class NPULogDecoder {
  // Parses the strings file to provide a decoder for parameterising the
  // string messages from the AIE tiles

private:
  std::string _elfstrings_file;
  std::map<int, std::string> _str_map;

  void parse_str_map() {
    std::ifstream file(_elfstrings_file);
    if (!file.is_open()) {
      std::cerr << "Error! unable to open the elfstrings file ("
                << _elfstrings_file << ")\n";
    }
    std::string line;
    while (std::getline(file, line)) {
      boost::char_separator<char> sep(",");
      boost::tokenizer<boost::char_separator<char>> tokens(line, sep);

      auto it = tokens.begin();
      if (it == tokens.end()) {
        // case where there are no tokens on the line
        continue;
      }
      int address;
      if (!(std::istringstream(*it) >> address)) {
        // Handle the case where the first token cannot be converted to an int
        continue;
      }
      ++it;
      if (it == tokens.end()) {
        // Handle the case where there's no second token
        continue;
      }
      std::string format_str = *it;
      _str_map[address] = format_str;
    }
  }

public:
  NPULogDecoder(std::string elfstrings_file)
      : _elfstrings_file(elfstrings_file) {
    parse_str_map();
  }

  // When given a string address return true if we have
  // the format string for it
  bool format_str_exists(uint32_t addr) {
    return _str_map.find(addr) != _str_map.end();
  }

  // Peel off a message payload from the start of the buffer
  // and render a format string with the parameters
  uint32_t *renderNextStr(std::vector<std::string> &log, uint32_t *buffer) {
    uint32_t straddr = *buffer;
    buffer++;
    if (format_str_exists(straddr)) {
      // Construct the string
      std::string frmt = _str_map[straddr];

      std::string out;
      for (std::string::size_type i = 0; i < frmt.size(); ++i) {
        if (frmt[i] == '%') {
          // We need to replace this and the next
          // char with an appropriately converted parameter
          i++;
          switch (frmt[i]) {
          case 'd': { // int type
            int intparam = *((int *)(buffer));
            out += std::to_string(intparam);
            buffer++;
            break;
          }
          case 'f': { // float type
            float floatparam = *((float *)(buffer));
            out += std::to_string(floatparam);
            buffer++;
            break;
          }
          case 'u': { // unsigned type
            unsigned unsignedparam = *((unsigned *)(buffer));
            out += std::to_string(unsignedparam);
            buffer++;
            break;
          }
          case 'x': { // hexadecimal type
            unsigned hexparam = *((unsigned *)(buffer));
            std::stringstream stream;
            stream << std::hex << hexparam;
            out += stream.str();
            buffer++;
            break;
          }
          }
        } else {
          out += frmt[i];
        }
      }

      log.emplace_back(out);
    }
    return buffer;
  }

  std::vector<std::string> decode(xrt::bo buffer) {
    uint32_t buffer_size = buffer.size();
    uint32_t *buffer_ptr = buffer.map<uint32_t *>();
    uint32_t *end_of_buffer = buffer_ptr + (buffer_size / sizeof(uint32_t));

    std::vector<std::string> rendered_log;
    while (buffer_ptr < end_of_buffer) {
      buffer_ptr = renderNextStr(rendered_log, buffer_ptr);
    }
    return rendered_log;
  }
};
