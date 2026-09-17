// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
// hrx_util.cpp - non-template implementation of the hrx util (JSON engine).
// The rest of hrx:: (format, ptree, optional, program_options, algorithm, uuid,
// xml) is header-inline in hrx_util.h.
#include "hrx_util.h"
#include <fstream>
#include <istream>
#include <iterator>
#include <ostream>

namespace hrx {
namespace property_tree {
namespace json_parser {

namespace detail {

struct Reader {
  const std::string &s;
  size_t i = 0;
  explicit Reader(const std::string &str) : s(str) {}

  void ws() {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
      ++i;
  }
  char peek() { return i < s.size() ? s[i] : '\0'; }

  [[noreturn]] void err(const std::string &m) { throw json_parser_error(m); }

  std::string parseString() {
    std::string out;
    ++i; // opening quote
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"')
        return out;
      if (c == '\\' && i < s.size()) {
        char e = s[i++];
        switch (e) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'n':
          out += '\n';
          break;
        case 't':
          out += '\t';
          break;
        case 'r':
          out += '\r';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'u': {
          if (i + 4 > s.size())
            err("bad \\u escape");
          unsigned code = (unsigned)std::stoul(s.substr(i, 4), nullptr, 16);
          i += 4;
          // minimal UTF-8 encode (BMP)
          if (code < 0x80)
            out += (char)code;
          else if (code < 0x800) {
            out += (char)(0xC0 | (code >> 6));
            out += (char)(0x80 | (code & 0x3F));
          } else {
            out += (char)(0xE0 | (code >> 12));
            out += (char)(0x80 | ((code >> 6) & 0x3F));
            out += (char)(0x80 | (code & 0x3F));
          }
          break;
        }
        default:
          out += e;
          break;
        }
      } else {
        out += c;
      }
    }
    err("unterminated string");
  }

  std::string parseLiteral() {
    size_t start = i;
    while (i < s.size()) {
      char c = s[i];
      if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' ||
          c == '\n' || c == '\r')
        break;
      ++i;
    }
    return s.substr(start, i - start);
  }

  void parseValue(ptree &node) {
    ws();
    char c = peek();
    if (c == '{')
      parseObject(node);
    else if (c == '[')
      parseArray(node);
    else if (c == '"')
      node.data() = parseString();
    else {
      std::string lit = parseLiteral();
      if (lit == "null")
        node.data() = ""; // boost stores null as empty
      else
        node.data() = lit; // numbers / true / false kept verbatim
    }
  }

  void parseObject(ptree &node) {
    ++i; // '{'
    ws();
    if (peek() == '}') {
      ++i;
      return;
    }
    while (true) {
      ws();
      if (peek() != '"')
        err("expected key string");
      std::string key = parseString();
      ws();
      if (peek() != ':')
        err("expected ':'");
      ++i;
      ptree child;
      parseValue(child);
      node.push_back({key, child});
      ws();
      char c = peek();
      if (c == ',') {
        ++i;
        continue;
      }
      if (c == '}') {
        ++i;
        break;
      }
      err("expected ',' or '}'");
    }
  }

  void parseArray(ptree &node) {
    ++i; // '['
    ws();
    if (peek() == ']') {
      ++i;
      return;
    }
    while (true) {
      ptree child;
      parseValue(child);
      node.push_back({"", child}); // boost: array elems have empty keys
      ws();
      char c = peek();
      if (c == ',') {
        ++i;
        continue;
      }
      if (c == ']') {
        ++i;
        break;
      }
      err("expected ',' or ']'");
    }
  }
};

bool isArray(const ptree &pt) {
  if (pt.empty())
    return false;
  for (auto &kv : pt)
    if (!kv.first.empty())
      return false;
  return true;
}

void writeString(std::ostream &os, const std::string &s) {
  os << '"';
  for (char c : s) {
    switch (c) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\t':
      os << "\\t";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\b':
      os << "\\b";
      break;
    case '\f':
      os << "\\f";
      break;
    default:
      os << c;
      break;
    }
  }
  os << '"';
}

void writeNode(std::ostream &os, const ptree &pt, bool pretty, int indent) {
  std::string pad(pretty ? indent * 4 : 0, ' ');
  std::string pad2(pretty ? (indent + 1) * 4 : 0, ' ');
  const char *nl = pretty ? "\n" : "";
  if (pt.empty()) {
    writeString(os, pt.data());
    return;
  }
  if (isArray(pt)) {
    os << '[' << nl;
    bool first = true;
    for (auto &kv : pt) {
      if (!first)
        os << ',' << nl;
      first = false;
      os << pad2;
      writeNode(os, kv.second, pretty, indent + 1);
    }
    os << nl << pad << ']';
  } else {
    os << '{' << nl;
    bool first = true;
    for (auto &kv : pt) {
      if (!first)
        os << ',' << nl;
      first = false;
      os << pad2;
      writeString(os, kv.first);
      os << (pretty ? ": " : ":");
      writeNode(os, kv.second, pretty, indent + 1);
    }
    os << nl << pad << '}';
  }
}

} // namespace detail

void read_json(std::istream &is, ptree &pt) {
  std::string content((std::istreambuf_iterator<char>(is)),
                      std::istreambuf_iterator<char>());
  detail::Reader r(content);
  pt.clear();
  r.parseValue(pt);
}

void read_json(const std::string &filename, ptree &pt) {
  std::ifstream f(filename, std::ios::binary);
  if (!f)
    throw json_parser_error("cannot open: " + filename);
  read_json(f, pt);
}

void write_json(std::ostream &os, const ptree &pt, bool pretty) {
  detail::writeNode(os, pt, pretty, 0);
  if (pretty)
    os << "\n";
}

void write_json(const std::string &filename, const ptree &pt, bool pretty) {
  std::ofstream f(filename, std::ios::binary);
  if (!f)
    throw json_parser_error("cannot write: " + filename);
  write_json(f, pt, pretty);
}

} // namespace json_parser
} // namespace property_tree
} // namespace hrx
