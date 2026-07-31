/******************************************************************************
* SPDX-License-Identifier: MIT
******************************************************************************/

/**
 * @file xpdi_compiler.h
 *
 * The one thing standard C cannot express: symbol visibility.
 *
 * Alignment does NOT belong here — the PDI headers use C11 `_Alignas`, which every
 * supported compiler accepts without a conditional. This header exists solely because
 * `__attribute__((visibility("default")))` has no ISO equivalent, and MSVC `cl` has no
 * `__attribute__` at all.
 *
 * Two axes, deliberately kept apart:
 *   - the TARGET: symbol visibility is an ELF/Mach-O concept. On PE/COFF there is
 *     nothing to express, so the macro is empty. That is why we test `_WIN32` rather
 *     than the compiler vendor — clang-cl targets PE too.
 *   - the COMPILER: a feature test (`__has_attribute`), not `defined(__GNUC__)`, so any
 *     compiler that supports the attribute gets it and any that does not still builds.
 *
 * `transformcdo` is a static library, so an empty expansion is correct on Windows — see
 * the `if(NOT MSVC)` guard around -fvisibility=default in ../src/CMakeLists.txt. If it
 * ever becomes a shared library, delete this header and use CMake's
 * generate_export_header(), which also handles __declspec(dllexport/dllimport).
 */

#ifndef XPDI_COMPILER_H
#define XPDI_COMPILER_H

#if !defined(_WIN32) && defined(__has_attribute)
#if __has_attribute(visibility)
#define XPDI_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifndef XPDI_EXPORT
#define XPDI_EXPORT
#endif

#endif /* XPDI_COMPILER_H */
