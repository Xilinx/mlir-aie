#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

// Helper function to generate random floating point numbers with high exponent
// variance (useful for blocked datatypes). Exponents are interpreted as base 2
inline float generateRandomFloatingPoint(std::mt19937 &eng, double minExp,
                                         double maxExp) {
  std::uniform_real_distribution<float> distrExp(minExp, maxExp);
  float exponent = distrExp(eng);

  std::uniform_real_distribution<float> distrMantissa(0.0, 1.0);
  float mantissa = distrMantissa(eng);
  
  return mantissa * std::pow(2.0, exponent);
}

// mbits - mantisa bits
// block - block size
// size  - length of the input array
// array - the array
// rounding - 0 for zero, 1 for nearest (tie to even)
// verbose - make some noise
// Quantization of an array of floats to bfp16.
// The input array is used as a scratchpad.
// The return array must be at least size * 1.125 and is structured as follows:
// 1. The first byte is the shared exponent (max exponent of the block).
// 2. The next *block* bytes are the quantized values.
inline void bfp16QuantFloat(int block, int size, float *array,
                            uint8_t *returnArray, int rounding, int verbose) {
  int mbits = 7;
  int start = 0, end, i, j, int8 = 1;
  unsigned int sign, exp, maxExp;
  unsigned int *p, mantissa, mask, value;
  int shift, maxShift;
  int8_t valueInt8; //, exp_buf[exp_buf_size];

  while (true) {
    // decide on the block (starting and ending point)
    end = start + block;
    if (end > size)
      end = size;

    // Find max exp
    maxExp = 0;
    for (i = start; i < end; i++) {
      p = (unsigned int *)(array + i);
      exp = *p >> 23;    // Get rid of mantissa
      exp &= 0x000000FF; // Keep the last 8 bit exponent (remove sign)

      if (maxExp < exp)
        maxExp = exp;
    }

    // Round each number
    maxShift = 0;
    for (i = start; i < end; i++) {
      p = (unsigned int *)(array + i);
      if (verbose) {
        printf("%d: value in float = %f\n", i, array[i]);
      }

      // sign, exp, and mantissa
      sign = *p & 0x80000000;     // Sign
      exp = *p >> 23;             // Get rid of mantissa
      exp &= 0x000000FF;          // Keep the last 8 bit exponent (remove sign)
      mantissa = *p & 0x007FFFFF; // 23-bit mantissa
      if (exp)
        mantissa |= 0x00800000; // add the implicit for normal value

      if (exp >= 255)
        continue; // Infinity or NaN remains

      // Calculate shift (bits needs to be zeroed out)
      // At least erase 23 - mbits + 1 (+1 is for making the implicit bit
      // explicit) or more if smaller
      shift = 23 - mbits + 1 + maxExp - exp;
      if (verbose) {
        printf("%d: shift=%d rounding=%d\n", i, shift, rounding);
        printf("%d: AS READ         sign=%d exp=%d mantissa=0x%08x\n", i, sign,
               exp, mantissa);
      }

      // Calculate rounding
      switch (rounding) {
      case 0:
        break; // do nothing, just truncate
      case 1:
        mantissa += 1 << (shift - 1); // add rounding for nearest
        mask = 1;
        for (j = 0; j <= shift; j++) {
          if (mantissa & mask) {
            if (j < shift)
              break; // some bit is set, not a tie case
            if (j == shift)
              mantissa &=
                  ~mask; // tie case, rounded to odd bits, adjust to even
          }
          mask <<= 1;
        }
        break;
      default:
        break;
      }
      if (verbose) {
        printf("%d: ADDED ROUNDING  sign=%d exp=%d mantissa=0x%08x\n", i, sign,
               exp, mantissa);
      }
      if (mantissa &
          0x01000000) { // rounding carried forward and need to adjust exponent
        if (exp < maxExp) { // This will not result in shifting of max_exp
          mantissa >>= 1;
          exp += 1;
          shift -= 1;
          if (exp >= 255)
            mantissa = 0; // overflow, signals infinity, should not happen
        } else {          // Keep the current scale and round down
          mantissa -= 1 << shift; // Round down instead
        }
      }
      if (verbose) {
        printf("%d: ADJUST CARRY    sign=%d exp=%d mantissa=0x%08x\n", i, sign,
               exp, mantissa);
      }

      // Perform shift
      if (shift < 32) {
        mantissa >>= shift; // setting bits to zero
        mantissa <<= shift;
      } else {
        mantissa = 0;
      }

      if (verbose) {
        printf("%d: SHIFTED         sign=%d exp=%d mantissa=0x%08x\n", i, sign,
               exp, mantissa);
      }
      if (mantissa) {
        if (shift < 32)
          valueInt8 = (sign >> 24) | (mantissa >> (17 + maxExp - exp));
        else
          valueInt8 = (sign >> 24);
        if (exp)
          mantissa &= ~0x00800000; // remove implicit bit for normal number
        value = sign | (exp << 23) | mantissa;
      } else {
        valueInt8 = 0;
        value = sign; // Mantissa is rounded to zero, signal zero
      }
      *p = value;
      if (verbose) {
        printf("%d: TO BE WRITTEN   sign=%d exp=%d mantissa=0x%08x\n", i, sign,
               exp, mantissa);
        printf("%d: value = %f\n", i, *(array + i));
        printf("%d: value_int8 = 0x%08x\n", int8, valueInt8);
        printf("max_exp = %d\n", maxExp);
      }
      returnArray[int8] = valueInt8;
      int8++;

      if (maxShift < shift)
        maxShift = shift;
    }
    returnArray[int8 - 9] = (uint8_t)maxExp;
    int8++;
    start = end;
    if (start >= size)
      break;
  }
}

// Convert a bfp16 array to a float.
// Assumes the return array is large enough to hold the result.
// Size should be the number of bytes in the input bfp16 array and returnArray should be at least size/1.125 in size.
inline void bfp16ebs8ToFloat(int size, uint8_t *array, float *returnArray,
                             int verbose) {
  int block = 8;
  int tempIndx = 0;
  for (int i = 0; i < size; i += block + 1) {
    uint8_t sharedExponent = (uint8_t)array[i];
    float multiplier;
    if (sharedExponent >= 127) {
      multiplier = 1.0 * (1 << (sharedExponent - 127));
    } else {
      multiplier = 1.0 / (1 << (127 - sharedExponent));
    }
    multiplier /= 64.0;
    if (verbose) {
      printf("shared_exponent = %d\n", sharedExponent);
      printf("multiplier = %f\n", multiplier);
    }
    for (int j = 1; j < block + 1; j++) {
      returnArray[tempIndx] = float(array[i + j] * multiplier);
      if (verbose) {
        printf("return_array[%d] = %f\n", tempIndx, returnArray[tempIndx]);
      }
      tempIndx++;
    }
  }
}

// Helper function to perform a matrix multiplication of two square matrices.
// Only meant for verification purposes
template <typename T>
inline void matrixMultiply(T *aIn, T *bIn, T *cOut, int size) {
  for (int i = 0; i < size * size; ++i) {
    cOut[i] = 0;
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        cOut[i * size + j] += aIn[i * size + k] * bIn[k * size + j];
      }
    }
  }
}