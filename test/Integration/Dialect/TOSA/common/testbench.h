#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdint.h>
#include <string>
#include <type_traits>

template <typename T>
struct Format;
template <>
struct Format<double> {
  static constexpr const char *value = "%la";
};
template <>
struct Format<float> {
  static constexpr const char *value = "%a";
};
template <>
struct Format<uint64_t> {
  static constexpr const char *value = "%llu";
};
template <>
struct Format<int64_t> {
  static constexpr const char *value = "%lld";
};
template <>
struct Format<uint32_t> {
  static constexpr const char *value = "%u";
};
template <>
struct Format<int32_t> {
  static constexpr const char *value = "%d";
};
template <>
struct Format<uint16_t> {
  static constexpr const char *value = "%hu";
};
template <>
struct Format<int16_t> {
  static constexpr const char *value = "%hd";
};
template <>
struct Format<uint8_t> {
  static constexpr const char *value = "%hhu";
};
template <>
struct Format<int8_t> {
  static constexpr const char *value = "%hhd";
};

template <typename T>
void writeItems(FILE *file, T const *data, unsigned num) {
  for (unsigned ind = 0; ind < num; ++ind) {
    if constexpr (std::is_same_v<T, bfloat16>) {
      fprintf(file, Format<float>::value, float(data[ind]));
    } else if constexpr (std::is_same_v<T, v2int4>) {
      int8_t tmp = *(uint8_t *)(data + ind);
      fprintf(file, Format<int8_t>::value, (tmp << 4) >> 4);
      fprintf(file, " ");
      fprintf(file, Format<int8_t>::value, tmp >> 4);
    } else if constexpr (std::is_same_v<T, v2uint4>) {
      uint8_t tmp = *(uint8_t *)(data + ind);
      fprintf(file, Format<uint8_t>::value, (tmp << 4) >> 4);
      fprintf(file, " ");
      fprintf(file, Format<uint8_t>::value, tmp >> 4);
    } else {
      fprintf(file, Format<T>::value, data[ind]);
    }
    fprintf(file, " ");
  }
}

template <>
void writeItems<cint32>(FILE *file, cint32 const *data, unsigned num) {
  writeItems<int32_t>(file, (int32_t const *)data, 2 * num);
}

template <>
void writeItems<cint16>(FILE *file, cint16 const *data, unsigned num) {
  writeItems<int16_t>(file, (int16_t const *)data, 2 * num);
}

#if __AIEARCH__ == 10
template <>
void writeItems<cfloat>(FILE *file, cfloat const *data, unsigned num) {
  writeItems<float>(file, (float const *)data, 2 * num);
}
#endif
template <typename T>
void writeData(T const *data, unsigned numWords, std::string const &filename) {
  FILE *file = fopen(filename.c_str(), "w");
  if (!file) {
    fprintf(stderr, "Failed to open %s for writing\n", filename.c_str());
    std::exit(1);
  }
  for (unsigned ind = 0; ind < numWords; ++ind) {
    writeItems<T>(file, data + ind, 1);
    fprintf(file, "\n");
  }
  if (file) {
    fclose(file);
    printf("Wrote %s\n", filename.c_str());
  }
}

template <typename T>
unsigned readItems(FILE *file, T *data, unsigned num) {
  unsigned numRead = 0;
  for (unsigned ind = 0; ind < num; ++ind) {
    if constexpr (std::is_same_v<T, bfloat16>) {
      float tmp;
      numRead += fscanf(file, Format<float>::value, &tmp);
      data[ind] = bfloat16(tmp);
    } else if constexpr (std::is_same_v<T, v2int4>) {
      int8_t tmp[2];
      numRead += fscanf(file, Format<int8_t>::value, &tmp[0]);
      numRead += fscanf(file, Format<int8_t>::value, &tmp[1]);
      data[ind] = (tmp[1] << 4) || (tmp[0] & 0xf);
    } else if constexpr (std::is_same_v<T, v2uint4>) {
      uint8_t tmp[2];
      numRead += fscanf(file, Format<uint8_t>::value, &tmp[0]);
      numRead += fscanf(file, Format<uint8_t>::value, &tmp[1]);
      data[ind] = (tmp[1] << 4) || (tmp[0] & 0xf);
    } else {
      numRead += fscanf(file, Format<T>::value, &data[ind]);
    }
  }
  if constexpr (std::is_same_v<T, v2int4> || std::is_same_v<T, v2uint4>) {
    numRead /= 2;
  }
  return numRead;
}

template <>
unsigned readItems<cint32>(FILE *file, cint32 *data, unsigned num) {
  return readItems<int32_t>(file, (int32_t *)data, num * 2) / 2;
}

template <>
unsigned readItems<cint16>(FILE *file, cint16 *data, unsigned num) {
  return readItems<int16_t>(file, (int16_t *)data, num * 2) / 2;
}

#if __AIEARCH__ == 10
template <>
unsigned readItems<cfloat>(FILE *file, cfloat *data, unsigned num) {
  return readItems<cfloat>(file, (cfloat *)data, num * 2) / 2;
}
#endif

template <typename T>
void readData(T *data, unsigned numWords, std::string const &filename) {
  FILE *file = fopen(filename.c_str(), "r");
  int ind = -1;
  bool ok = file;
  for (++ind; ok && ind < numWords; ++ind) {
    ok = 1 == readItems<T>(file, data + ind, 1);
  }
  if (file)
    fclose(file);
  if (!ok) {
    fprintf(stderr, "Failed while reading %s at item %d\n", filename.c_str(),
            ind);
    std::exit(1);
  }
  printf("Read %s\n", filename.c_str());
}

bool almostEqual(float val1, float val2, float relTol, float absTol) {
  return std::fabs(val1 - val2) <=
         std::max(relTol * std::max(std::fabs(val1), std::fabs(val2)), absTol);
}

template <typename T>
bool almostEqual(T val1, T val2, int absTol) {
  return (val1 < val2) ? (val2 - val1 <= absTol) : (val1 - val2 <= absTol);
}

template <typename T>
class CheckData {
public:
  CheckData(unsigned absTol, float relTol, float absTolF)
      : _absTol(absTol), _relTol(relTol), _absTolF(absTolF), _numErrors(0),
        _maxNumToReport(10) {}

  unsigned check(T const *data, T const *expected, unsigned num) {
    for (unsigned ind = 0; ind < num; ++ind) {
      checkItem(data[ind], expected[ind], ind);
    }
    if (_numErrors == 0) {
      printf("PASS\n");
    } else {
      printf("FAIL: %u mismatches\n", _numErrors);
    }
    return _numErrors;
  }

private:
  void checkItem(T const &data, T const &expected, unsigned ind) {
    bool ok;
    if constexpr (std::is_same_v<T, float>) {
      ok = almostEqual(data, expected, _relTol, _absTolF);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
      ok = almostEqual(float(data), float(expected), _relTol, _absTolF);
    } else if constexpr (std::is_same_v<T, v2int4>) {
      int tmpD = (int8_t const &)(data);
      int tmpE = (int8_t const &)(expected);
      ok = almostEqual<int>((tmpD << 4) >> 4, (tmpE << 4) >> 4, _absTol);
      ok &= almostEqual<int>(tmpD >> 4, tmpE >> 4, _absTol);
    } else if constexpr (std::is_same_v<T, v2uint4>) {
      unsigned tmpD = (uint8_t const &)(data);
      unsigned tmpE = (uint8_t const &)(expected);
      ok = almostEqual<unsigned>((tmpD << 4) >> 4, (tmpE << 4) >> 4, _absTol);
      ok &= almostEqual<unsigned>(tmpD >> 4, tmpE >> 4, _absTol);
    } else {
      ok = almostEqual(data, expected, _absTol);
    }
    if (!ok && _numErrors < _maxNumToReport) {
      printf("Mismatch at item %u: ", ind);
      printf(" expected: ");
      writeItems(stdout, &expected, 1);
      printf(" but got: ");
      writeItems(stdout, &data, 1);
      printf("\n");
    }
    if (!ok)
      _numErrors++;
  }
  int _absTol;
  float _relTol;
  float _absTolF;
  unsigned _numErrors;
  unsigned _maxNumToReport;
};

template <typename T>
bool checkData(T const *data, T const *expected, unsigned num,
               unsigned absTol = 0, float relTol = 0.0, float absTolF = 0.0) {
  return 0 == CheckData<T>(absTol, relTol, absTolF).check(data, expected, num);
}

template <>
bool checkData<cint16>(cint16 const *data, cint16 const *expected, unsigned num,
                       unsigned absTol, float relTol, float absTolF) {
  return checkData<int16_t>((int16_t const *)data, (int16_t const *)expected,
                            2 * num, absTol, relTol, absTolF);
}

template <>
bool checkData<cint32>(cint32 const *data, cint32 const *expected, unsigned num,
                       unsigned absTol, float relTol, float absTolF) {
  return checkData<int32_t>((int32_t const *)data, (int32_t const *)expected,
                            2 * num, absTol, relTol, absTolF);
}

inline void reportCycleCount(int count, std::string const &filename) {
#ifdef __chess__
  printf("Cycle count: %d\n", count);
  if (FILE *file = fopen(filename.c_str(), "w")) {
    fprintf(file, "%d\n", count);
    fclose(file);
  }
#endif
}

// Generate a random integer in range [0,2^bits) for unsigned types and
// [-2^(bits-1),2^(bits-1)) for signed types.
template <typename T>
T random_integer(unsigned int bits = 8 * sizeof(T)) {
  // chess run-time library has RAND_MAX=2^15 -1
  constexpr int rbits = 15; // 32 - __builtin_ctz (uint32_t(RAND_MAX));

  uint64_t val = 0;
  for (int k = 0; k < (64 + rbits - 1) / rbits; ++k) {
    val <<= rbits;
    val |= ((uint32_t)rand()) & ((1 << rbits) - 1u);
  }
  using TT = typename std::conditional<std::is_signed<T>::value, int64_t,
                                       uint64_t>::type;
  TT sval = (TT)val;
  if (bits < 64) {
    sval <<= (64 - bits);
    sval >>= (64 - bits);
  }
  return (T)(sval);
}

float random_float(int minExp = -126, int maxExp = 127,
                   unsigned mantBits = 23) {
  unsigned expRange;
  if (minExp >= -126 && maxExp <= 127 && minExp <= maxExp) {
    expRange = 1 + maxExp - minExp;
  } else {
    expRange = 255;
    minExp = -126;
  }
  unsigned exponent = 127 + (int(rand() % expRange) + minExp);
  unsigned mant = random_integer<unsigned>();

  if (mantBits < 23 && mantBits > 0) {
    // Set least significant bits to 0.
    mant <<= 23 - mantBits;
  }

  unsigned val = (mant & (~(0u) >> 9)) | (exponent << 23);
  if (rand() & 0x1)
    val |= 1u << 31;
  else
    val &= ~(1u << 31);
  return *(float *)&val;
}

bfloat16 random_bfloat16(int minExp = -126, int maxExp = 127,
                         unsigned mantBits = 7) {
  unsigned expRange;
  if (minExp >= -126 && maxExp <= 127 && minExp <= maxExp) {
    expRange = 1 + maxExp - minExp;
  } else {
    expRange = 255;
    minExp = -126;
  }
  unsigned exponent = 127 + (int(rand() % expRange) + minExp);
  unsigned short mant = (unsigned short)(rand());

  if (mantBits < 7 && mantBits > 0) {
    // Set least significant bits to 0.
    mant <<= 7 - mantBits;
  }
  unsigned short val = (mant & (~(0u) >> 25)) | (exponent << 7);
  if (rand() & 0x1)
    val |= 1u << 15;
  else
    val &= ~(1u << 15);
  return *(bfloat16 *)&val;
}

#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)
#ifndef DATA_DIR
#define DATA_DIR data
#endif
