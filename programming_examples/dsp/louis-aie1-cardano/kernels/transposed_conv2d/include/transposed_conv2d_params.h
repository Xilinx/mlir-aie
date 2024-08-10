#ifndef CONV2D_PARAMS_H
#define CONV2D_PARAMS_H

#define MR (((M%2) == 1) ? M + 1 : M)

#define CinUp (((Cin+7)/8) * 8)
#define CoutUp (((Cout+7)/8) * 8)

//#define outHeight (S * (N - 1) + F - 2 * P)
//#define outWidthR (S * (M - 1) + F - 2 * P)

#define outHeight (S * (N - 1) + F)
#define outWidthR (S * (M - 1) + F)

#define outHeightInner ((N - F) / S + 1)
#define outWidthInnerR ((M - F) / S + 1)
#define outWidthInner ((outWidthR%2) == 1 ? outWidthR + 1 : outWidthR)

#define outWidth ((outWidthR%2) == 1 ? outWidthR + 1 : outWidthR)

#define SHIFT 0

#define inTileSize MR * N * CinUp
#define outTileSize outWidth * outHeight * CoutUp
#define weightTileSize CoutUp * CinUp * F * F
#define weightSize 8 * F * F * CinUp

#define AIn_FILENAME "../data/AIn.txt"
#define WIn_FILENAME "../data/WIn.txt"
#define AOutRef_FILENAME "../data/AOutRef.txt"
#define AOutRefReg_FILENAME "../data/AOut.txt"

#define realOutTileSize outHeight * outWidthR * Cout
#define realInTileSize M * N * CinUp

#endif
