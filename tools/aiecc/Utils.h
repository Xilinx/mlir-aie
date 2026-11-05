//===- Utils.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Domain-agnostic helpers used across aiecc — currently filesystem-path
// mangling. Nothing in here should know about MLIR/AIE types.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_UTILS_H
#define AIECC_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace xilinx::aiecc {

// Serializes diagnostic/progress writes to the shared stdout/stderr so that
// concurrent worker threads (and the tool-invocation echo) don't interleave
// their lines. Any code that prints a full log line from a worker thread should
// hold this across the write (and flush before releasing).
inline std::mutex &logMutex() {
  static std::mutex m;
  return m;
}

// RAII: while alive (when `enable`), redirect the process's stdout+stderr
// (fd 1 & 2) into a temp file, capturing an in-process tool library's chatter
// instead of letting it leak (and corrupt the single-line --progress display).
// On destruction it restores the fds and loads the captured text into `out`, so
// the caller can choose to replay it -- e.g. only when the library call failed.
// A disabled instance (verbose runs) is a no-op and leaves `out` empty, letting
// the library write straight to the terminal.
//
// It redirects *process-global* fds, so it holds `logMutex()` while they are
// redirected: worker threads running alongside it write their progress and
// failure output under that lock, and would otherwise write into the capture.
class CaptureStdio {
public:
  CaptureStdio(bool enable, std::string &out) : out(out) {
#ifndef _WIN32
    if (!enable)
      return;
    llvm::SmallString<128> path;
    int fd = -1;
    if (llvm::sys::fs::createTemporaryFile("aiecc-lib", "log", fd, path))
      return;
    tmpPath = std::string(path.str());
    log = std::unique_lock<std::mutex>(logMutex());
    std::fflush(stdout);
    std::fflush(stderr);
    savedOut = ::dup(STDOUT_FILENO);
    savedErr = ::dup(STDERR_FILENO);
    ::dup2(fd, STDOUT_FILENO);
    ::dup2(fd, STDERR_FILENO);
    ::close(fd);
#else
    (void)enable;
#endif
  }
  ~CaptureStdio() {
#ifndef _WIN32
    if (savedOut < 0)
      return;
    std::fflush(stdout);
    std::fflush(stderr);
    ::dup2(savedOut, STDOUT_FILENO);
    ::close(savedOut);
    if (savedErr >= 0) {
      ::dup2(savedErr, STDERR_FILENO);
      ::close(savedErr);
    }
    if (auto buf = llvm::MemoryBuffer::getFile(tmpPath))
      out = (*buf)->getBuffer().str();
    llvm::sys::fs::remove(tmpPath);
#endif
  }
  CaptureStdio(const CaptureStdio &) = delete;
  CaptureStdio &operator=(const CaptureStdio &) = delete;

private:
  std::string &out;
  int savedOut = -1, savedErr = -1;
  std::string tmpPath;
  // Released after the destructor body has restored the fds.
  std::unique_lock<std::mutex> log;
};

// While aiecc runs an attempt it may still discard, everything the process
// writes to stdout and stderr (its own lines, diagnostics, the tools it runs)
// goes to files instead; the attempt that stands has it copied out, in order,
// and the rest of the build prints live. Two streams on one file share one
// held file, so their order is kept. Like CaptureStdio, POSIX only: elsewhere
// nothing is held. One instance, so that a signal handler can find it.
class OutputHold {
public:
  static OutputHold &get() {
    static OutputHold hold;
    return hold;
  }

  /// What an attempt wrote to each stream; on one file, all of it is `err`.
  struct Held {
    std::string out, err;
  };

  /// Start holding. Call with logMutex() held.
  bool open() {
#ifndef _WIN32
    drainLate();
    llvm::outs().flush();
    llvm::errs().flush();
    std::fflush(stdout);
    std::fflush(stderr);
    struct stat out, err;
    bool shared = ::fstat(STDOUT_FILENO, &out) == 0 &&
                  ::fstat(STDERR_FILENO, &err) == 0 &&
                  out.st_dev == err.st_dev && out.st_ino == err.st_ino;
    if (!streams[1].open(STDERR_FILENO, -1))
      return false;
    if (!streams[0].open(STDOUT_FILENO, shared ? streams[1].heldFd : -1)) {
      streams[1].stop();
      return false;
    }
    holding = true;
    return true;
#else
    return false;
#endif
  }

  /// Stop holding and print what was held. Call with logMutex() held.
  /// `endLine` ends an open status line on the terminal first; a held
  /// newline that ended it back then is dropped.
  void release(bool endLine) {
    Held held = take();
    llvm::outs() << held.out;
    llvm::outs().flush();
    llvm::StringRef err(held.err);
    if (endLine && !err.empty()) {
      llvm::errs() << '\n';
      err.consume_front("\n");
    }
    llvm::errs() << err;
    llvm::errs().flush();
  }

  /// Stop holding and hand back what was held, unprinted. Call with
  /// logMutex() held.
  Held take() {
#ifndef _WIN32
    if (!holding)
      return {};
    llvm::outs().flush();
    llvm::errs().flush();
    std::fflush(stdout);
    std::fflush(stderr);
    holding = false;
    for (Stream &stream : streams)
      stream.stop();
    Held held;
    held.out = streams[0].read();
    held.err = streams[1].read();
    return held;
#else
    return {};
#endif
  }

  /// Where the status line goes: the terminal, also while holding.
  llvm::raw_ostream &terminal() {
#ifndef _WIN32
    if (holding) {
      if (!terminalStream || terminalStreamFd != streams[1].terminalFd) {
        terminalStream = std::make_unique<llvm::raw_fd_ostream>(
            streams[1].terminalFd, /*shouldClose=*/false, /*unbuffered=*/true);
        terminalStreamFd = streams[1].terminalFd;
      }
      return *terminalStream;
    }
#endif
    return llvm::errs();
  }

  /// Print what is held, and what a tool still running wrote after it was
  /// released. Async-signal-safe: a crash or an interrupt calls it.
  void flushOnAbort() {
#ifndef _WIN32
    for (Stream &stream : streams)
      stream.copyOut();
#endif
  }

  /// Flush what is held on a crash (the handlers LLVM runs for fault
  /// signals, report_fatal_error's abort included) and on an interrupt or
  /// termination, which then proceeds as it would have. Call once, after LLVM
  /// installed its handlers.
  static void flushOnSignals() {
#ifndef _WIN32
    llvm::sys::AddSignalHandler([](void *) { get().flushOnAbort(); }, nullptr);
    for (int sig : {SIGHUP, SIGINT, SIGTERM}) {
      struct sigaction action = {};
      action.sa_handler = [](int sig) {
        get().flushOnAbort();
        ::sigaction(sig, &previousAction(sig), nullptr);
        ::raise(sig);
      };
      sigemptyset(&action.sa_mask);
      ::sigaction(sig, &action, &previousAction(sig));
    }
#endif
  }

  /// Print what tools still running wrote after the release, and close the
  /// held files. Call with logMutex() held.
  void drainLate() {
#ifndef _WIN32
    if (holding)
      return;
    llvm::outs().flush();
    llvm::errs().flush();
    for (Stream &stream : streams) {
      stream.copyOut();
      stream.close();
    }
#endif
  }

private:
#ifndef _WIN32
  static struct sigaction &previousAction(int sig) {
    static struct sigaction previous[NSIG];
    return previous[sig];
  }

  struct Stream {
    volatile int fd = -1;
    volatile int heldFd = -1;
    volatile int terminalFd = -1;
    volatile off_t copied = 0;
    volatile bool owner = false;
    volatile bool holding = false;

    // Hold `stream` in a new file, or in `sharedFd` when not -1.
    bool open(int stream, int sharedFd) {
      int held = sharedFd;
      if (held < 0) {
        llvm::SmallString<128> path;
        if (llvm::sys::fs::createTemporaryFile("aiecc-held", "log", held, path))
          return false;
        // The open descriptor keeps the file; nothing is left on a kill.
        llvm::sys::fs::remove(path);
      }
      fd = stream;
      heldFd = held;
      owner = sharedFd < 0;
      copied = 0;
      terminalFd = ::dup(stream);
      ::dup2(held, stream);
      holding = true;
      return true;
    }

    void stop() {
      if (!holding)
        return;
      ::dup2(terminalFd, fd);
      ::close(terminalFd);
      terminalFd = fd;
      holding = false;
    }

    std::string read() {
      std::string text;
      if (!owner || heldFd < 0)
        return text;
      char buf[4096];
      for (ssize_t n; (n = ::pread(heldFd, buf, sizeof(buf), copied)) > 0;
           copied += n)
        text.append(buf, n);
      return text;
    }

    void copyOut() {
      if (!owner || heldFd < 0)
        return;
      int out = holding ? terminalFd : fd;
      char buf[4096];
      off_t at = copied;
      for (ssize_t n; (n = ::pread(heldFd, buf, sizeof(buf), at)) > 0; at += n)
        for (ssize_t done = 0; done < n;) {
          ssize_t w = ::write(out, buf + done, n - done);
          if (w <= 0)
            return;
          done += w;
        }
      copied = at;
    }

    void close() {
      if (owner && heldFd >= 0)
        ::close(heldFd);
      heldFd = -1;
      owner = false;
    }
  };
  Stream streams[2];
#endif
  volatile bool holding = false;
  std::unique_ptr<llvm::raw_fd_ostream> terminalStream;
  int terminalStreamFd = -1;
};

// Reinterpret a u32 instruction stream as its raw little-endian bytes. Every
// AIE binary translator emits `std::vector<uint32_t>`, but artifacts are
// materialized as bytes, so this conversion is shared by all of them.
inline std::vector<char> wordsToBytes(llvm::ArrayRef<uint32_t> words) {
  const auto *p = reinterpret_cast<const char *>(words.data());
  return std::vector<char>(p, p + words.size() * sizeof(uint32_t));
}

// Absolutize a path against the current working directory.
inline std::string absolutePath(llvm::StringRef p) {
  llvm::SmallString<256> abs(p);
  llvm::sys::fs::make_absolute(abs);
  return std::string(abs);
}

// Recursively copy the contents of directory `src` into `dst` (creating `dst`).
// Used as the cross-device fallback for publishDirectory and to snapshot a
// directory artifact into a checkpoint.
inline std::error_code copyDirectoryRecursively(llvm::StringRef src,
                                                llvm::StringRef dst) {
  if (std::error_code ec = llvm::sys::fs::create_directories(dst))
    return ec;
  std::error_code ec;
  for (llvm::sys::fs::recursive_directory_iterator it(src, ec), end;
       it != end && !ec; it.increment(ec)) {
    llvm::StringRef rel = llvm::StringRef(it->path()).substr(src.size());
    while (rel.starts_with("/"))
      rel = rel.drop_front();
    llvm::SmallString<256> target(dst);
    llvm::sys::path::append(target, rel);
    if (llvm::sys::fs::is_directory(it->path())) {
      if ((ec = llvm::sys::fs::create_directories(target)))
        break;
    } else if ((ec = llvm::sys::fs::copy_file(it->path(), target))) {
      break;
    }
  }
  return ec;
}

// Resolve a (possibly relative) path against, in order: cwd, workDir, and the
// directory containing inputFile. The first candidate that exists wins; if
// none exist, the inputFile-relative path is returned as a best-effort guess.
// Absolute inputs pass through unchanged.
inline std::string resolveExternalPath(llvm::StringRef rel,
                                       llvm::StringRef inputFile,
                                       llvm::StringRef workDir) {
  if (llvm::sys::path::is_absolute(rel))
    return rel.str();

  llvm::SmallString<256> cwd;
  llvm::sys::fs::current_path(cwd);
  llvm::sys::path::append(cwd, rel);
  if (llvm::sys::fs::exists(cwd))
    return std::string(cwd);

  if (!workDir.empty()) {
    llvm::SmallString<256> w(workDir);
    llvm::sys::path::append(w, rel);
    if (llvm::sys::fs::exists(w))
      return std::string(w);
  }

  if (!inputFile.empty()) {
    llvm::SmallString<256> inDir = llvm::sys::path::parent_path(inputFile);
    if (inDir.empty())
      llvm::sys::fs::current_path(inDir);
    llvm::sys::path::append(inDir, rel);
    llvm::sys::path::remove_dots(inDir, /*remove_dot_dot=*/true);
    return std::string(inDir);
  }
  return rel.str();
}

// Extract the object-file paths referenced by a BCF's `_include _file <path>`
// directives. The chess linker needs these objects passed on its command line
// (the BCF directive alone does not pull them into the link), so the xbridge
// link step resolves and forwards each one.
inline std::vector<std::string> parseBcfIncludeFiles(llvm::StringRef bcf) {
  std::vector<std::string> files;
  llvm::SmallVector<llvm::StringRef> lines;
  bcf.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.consume_front("_include _file ")) {
      llvm::StringRef path = line.trim();
      if (!path.empty())
        files.push_back(path.str());
    }
  }
  return files;
}

// Discover the aietools (Vitis AIE / Chess) install directory. Search order:
//   1. an explicit override (e.g. the --aietools flag),
//   2. the $AIETOOLS_ROOT environment variable,
//   3. derive it from `xchesscc` on PATH — it lives at <aietools>/bin/xchesscc
//      or <aietools>/bin/unwrapped/lnx64.o/xchesscc.
// Returns "" if none of these resolve to a directory.
inline std::string discoverAietoolsDir(llvm::StringRef overrideDir = "") {
  if (!overrideDir.empty() && llvm::sys::fs::is_directory(overrideDir))
    return overrideDir.str();
  if (const char *env = std::getenv("AIETOOLS_ROOT"))
    if (llvm::sys::fs::is_directory(env))
      return std::string(env);
  auto xchesscc = llvm::sys::findProgramByName("xchesscc");
  if (xchesscc) {
    llvm::StringRef binDir = llvm::sys::path::parent_path(*xchesscc);
    // Unwrap <aietools>/bin/unwrapped/lnx64.o/xchesscc back to <aietools>/bin.
    if (llvm::sys::path::filename(binDir) == "lnx64.o")
      binDir = llvm::sys::path::parent_path(
          llvm::sys::path::parent_path(binDir)); // up from lnx64.o/unwrapped
    return std::string(llvm::sys::path::parent_path(binDir)); // up from bin
  }
  return "";
}

// Install prefix of the running executable (parent of bin/), used to locate
// aie_runtime_lib. Falls back to a sibling install/ dir, then
// $MLIR_AIE_INSTALL_DIR, then the computed prefix.
inline std::string getInstallDir() {
  std::string mainExe = llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getInstallDir));
  llvm::SmallString<256> prefix(
      llvm::sys::path::parent_path(llvm::sys::path::parent_path(mainExe)));
  auto hasRuntimeLib = [](llvm::StringRef dir) {
    llvm::SmallString<256> p(dir);
    llvm::sys::path::append(p, "aie_runtime_lib");
    return llvm::sys::fs::is_directory(p);
  };
  if (hasRuntimeLib(prefix))
    return std::string(prefix);
  llvm::SmallString<256> sibling(llvm::sys::path::parent_path(prefix));
  llvm::sys::path::append(sibling, "install");
  if (hasRuntimeLib(sibling))
    return std::string(sibling);
  if (const char *env = std::getenv("MLIR_AIE_INSTALL_DIR"))
    if (llvm::sys::fs::is_directory(env))
      return std::string(env);
  return std::string(prefix);
}

} // namespace xilinx::aiecc

#endif // AIECC_UTILS_H
