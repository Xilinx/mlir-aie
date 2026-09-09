//===- RegionMap.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/RegionMap.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

using namespace aiesim;

namespace {

bool isIdentChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.';
}

/// Strip /* */ comments. The scripts carry them mid-statement (`. += 0xD00;
/// /* stack */`), so a line-oriented reader that ignored them would still
/// work, but only by luck.
std::string stripComments(const std::string &text) {
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t close = text.find("*/", i + 2);
      if (close == std::string::npos)
        break;
      i = close + 2;
      out += ' ';
      continue;
    }
    out += text[i++];
  }
  return out;
}

/// A numeric literal, hex or decimal. Linker scripts also accept K/M suffixes;
/// AIETargetLdScript.cpp emits neither, so they are not accepted here rather
/// than half-supported.
bool parseNumber(const std::string &s, size_t &i, uint64_t &out) {
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
  size_t start = i;
  int base = 10;
  if (s.compare(i, 2, "0x") == 0 || s.compare(i, 2, "0X") == 0) {
    base = 16;
    i += 2;
  }
  size_t digits = i;
  while (i < s.size() && std::isxdigit(static_cast<unsigned char>(s[i]))) {
    if (base == 10 && !std::isdigit(static_cast<unsigned char>(s[i])))
      break;
    ++i;
  }
  if (i == digits) {
    i = start;
    return false;
  }
  out = std::strtoull(s.substr(digits, i - digits).c_str(), nullptr, base);
  return true;
}

void skipSpace(const std::string &s, size_t &i) {
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
}

bool literal(const std::string &s, size_t &i, const char *lit) {
  skipSpace(s, i);
  size_t n = std::char_traits<char>::length(lit);
  if (s.compare(i, n, lit) != 0)
    return false;
  i += n;
  return true;
}

std::string readIdent(const std::string &s, size_t &i) {
  skipSpace(s, i);
  size_t start = i;
  while (i < s.size() && isIdentChar(s[i]))
    ++i;
  return s.substr(start, i - start);
}

/// The MEMORY block: `name (ATTRS) : ORIGIN = n, LENGTH = n`.
void parseMemoryBlock(const std::string &s, RegionMap &out) {
  size_t open = s.find("MEMORY");
  if (open == std::string::npos)
    return;
  open = s.find('{', open);
  size_t close = s.find('}', open == std::string::npos ? 0 : open);
  if (open == std::string::npos || close == std::string::npos)
    return;

  std::string body = s.substr(open + 1, close - open - 1);
  size_t i = 0;
  while (i < body.size()) {
    std::string name = readIdent(body, i);
    if (name.empty()) {
      ++i;
      continue;
    }
    size_t save = i;
    if (!literal(body, i, "(")) {
      i = save + 1;
      continue;
    }
    while (i < body.size() && body[i] != ')')
      ++i;
    if (i < body.size())
      ++i;
    if (!literal(body, i, ":") || !literal(body, i, "ORIGIN") ||
        !literal(body, i, "="))
      continue;
    uint64_t origin = 0, length = 0;
    if (!parseNumber(body, i, origin))
      continue;
    if (!literal(body, i, ",") || !literal(body, i, "LENGTH") ||
        !literal(body, i, "="))
      continue;
    if (!parseNumber(body, i, length))
      continue;

    Region r;
    r.name = name;
    r.begin = static_cast<uint32_t>(origin);
    r.size = static_cast<uint32_t>(length);
    r.kind = name == "program" ? RegionKind::Program : RegionKind::Data;
    out.add(std::move(r));
  }
}

bool looksLikeStack(const std::string &name) {
  // AIETargetLdScript.cpp emits `_sp_start_value_DM_stack`. Match on the
  // stack substring rather than the exact spelling so a renamed symbol
  // degrades to "found it" instead of "no stack in this design", which would
  // silently disable the guard.
  std::string lower;
  for (char c : name)
    lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return lower.find("stack") != std::string::npos ||
         lower.find("_sp_start_value") != std::string::npos;
}

} // namespace

const char *aiesim::regionKindName(RegionKind kind) {
  switch (kind) {
  case RegionKind::Program: return "program";
  case RegionKind::Data: return "data";
  case RegionKind::Stack: return "stack";
  case RegionKind::Buffer: break;
  }
  return "buffer";
}

void RegionMap::add(Region r) {
  auto at = std::lower_bound(items.begin(), items.end(), r,
                             [](const Region &a, const Region &b) {
                               if (a.begin != b.begin)
                                 return a.begin < b.begin;
                               return a.name < b.name;
                             });
  items.insert(at, std::move(r));
}

const Region *RegionMap::stack() const {
  for (const Region &r : items)
    if (r.kind == RegionKind::Stack)
      return &r;
  return nullptr;
}

const Region *RegionMap::findContaining(uint32_t addr) const {
  // Prefer the most specific match: the MEMORY `data` region spans the named
  // allocations inside it, so a plain first-hit would always answer "data".
  const Region *best = nullptr;
  for (const Region &r : items)
    if (r.contains(addr))
      if (!best || r.size < best->size)
        best = &r;
  return best;
}

bool RegionMap::stackClearance(uint32_t &bytesOut,
                               std::string &nextRegionOut) const {
  const Region *s = stack();
  if (!s)
    return false;
  const Region *next = nullptr;
  for (const Region &r : items) {
    if (r.kind == RegionKind::Program || r.kind == RegionKind::Data)
      continue; // Containers, not neighbours.
    if (&r == s || r.begin < s->end())
      continue;
    if (!next || r.begin < next->begin)
      next = &r;
  }
  if (!next)
    return false;
  bytesOut = next->begin - s->end();
  nextRegionOut = next->name;
  return true;
}

std::vector<RegionMap::Overlap> RegionMap::overlaps() const {
  std::vector<Overlap> found;
  for (size_t i = 0; i < items.size(); ++i) {
    const Region &a = items[i];
    if (a.kind == RegionKind::Program || a.kind == RegionKind::Data)
      continue;
    for (size_t j = i + 1; j < items.size(); ++j) {
      const Region &b = items[j];
      if (b.kind == RegionKind::Program || b.kind == RegionKind::Data)
        continue;
      if (b.begin >= a.end())
        break; // Sorted, so nothing further can intersect a.
      found.push_back({a.name, b.name, b.begin, std::min(a.end(), b.end())});
    }
  }
  return found;
}

bool RegionMap::checkStackPointer(uint32_t sp, std::string &why) const {
  const Region *s = stack();
  if (!s)
    return true; // No reservation described, so nothing to check against.
  if (s->contains(sp))
    return true;

  char buf[256];
  if (sp >= s->end()) {
    const Region *hit = findContaining(sp);
    std::snprintf(buf, sizeof(buf),
                  "stack pointer 0x%x is past the end of '%s' "
                  "[0x%x, 0x%x); it is now inside '%s'",
                  sp, s->name.c_str(), s->begin, s->end(),
                  hit ? hit->name.c_str() : "unallocated memory");
  } else {
    std::snprintf(buf, sizeof(buf),
                  "stack pointer 0x%x is below the start of '%s' [0x%x, 0x%x)",
                  sp, s->name.c_str(), s->begin, s->end());
  }
  why = buf;
  return false;
}

bool RegionMap::checkWrite(uint32_t addr, uint32_t len, std::string &why) const {
  if (items.empty() || !len)
    return true;
  const Region *owner = findContaining(addr);
  // An address in no region at all is not this check's business: plenty of
  // data memory is legitimately unallocated by the script, and faulting on it
  // would make the guard unusable.
  if (!owner || !owner->escapes(addr, len))
    return true;

  const Region *next = nullptr;
  for (const Region &r : items) {
    if (r.kind == RegionKind::Program || r.kind == RegionKind::Data)
      continue;
    if (r.begin >= owner->end() && (!next || r.begin < next->begin))
      next = &r;
  }
  char buf[288];
  std::snprintf(buf, sizeof(buf),
                "write of %u byte(s) at 0x%x starts in '%s' [0x%x, 0x%x) and "
                "runs %llu byte(s) past its end, into '%s'",
                len, addr, owner->name.c_str(), owner->begin, owner->end(),
                (unsigned long long)(uint64_t(addr) + len - owner->end()),
                next ? next->name.c_str() : "unallocated memory");
  why = buf;
  return false;
}

bool aiesim::parseLinkerScript(const std::string &text, RegionMap &out,
                               std::string &error) {
  if (text.find("MEMORY") == std::string::npos &&
      text.find("SECTIONS") == std::string::npos) {
    error = "not a linker script: no MEMORY or SECTIONS block";
    return false;
  }

  std::string s = stripComments(text);
  parseMemoryBlock(s, out);

  // Walk the location-counter statements. The emitted shape is
  //   . = <addr>;  [<sym> = .;]  . += <size>;
  // where the symbol is absent for the padding runs that skip a neighbour
  // band, so a symbol is optional and its absence just means "no region here".
  uint64_t cur = 0;
  std::string pending;
  size_t i = 0;
  while (i < s.size()) {
    skipSpace(s, i);
    if (i >= s.size())
      break;

    if (s[i] == '.' && (i + 1 >= s.size() || !isIdentChar(s[i + 1]))) {
      size_t save = i;
      ++i;
      skipSpace(s, i);
      if (s.compare(i, 2, "+=") == 0) {
        i += 2;
        uint64_t size = 0;
        if (parseNumber(s, i, size)) {
          if (!pending.empty()) {
            Region r;
            r.name = pending;
            r.begin = static_cast<uint32_t>(cur);
            r.size = static_cast<uint32_t>(size);
            r.kind = looksLikeStack(pending) ? RegionKind::Stack
                                             : RegionKind::Buffer;
            out.add(std::move(r));
            pending.clear();
          }
          cur += size;
          continue;
        }
      } else if (i < s.size() && s[i] == '=') {
        ++i;
        uint64_t addr = 0;
        if (parseNumber(s, i, addr)) {
          cur = addr;
          // A new location counter abandons an unterminated symbol rather
          // than attaching it to the wrong address.
          pending.clear();
          continue;
        }
      }
      i = save + 1;
      continue;
    }

    if (isIdentChar(s[i])) {
      size_t save = i;
      std::string ident = readIdent(s, i);
      size_t j = i;
      if (literal(s, j, "=") && literal(s, j, ".") && literal(s, j, ";")) {
        pending = ident;
        i = j;
        continue;
      }
      i = save + ident.size();
      if (i == save)
        ++i;
      continue;
    }
    ++i;
  }
  return true;
}
