#ifndef CONV2D_PARAMS_H
#define CONV2D_PARAMS_H

#define MR (((M%2) == 1) ? M + 1 : M)

#define CinUp (((Cin+7)/8) * 8)
#define CoutUp (((Cout+7)/8) * 8)

#define outHeight ((N - F + 2*P) / S + 1)
#define outWidthR ((M - F + 2*P) / S + 1)

#define outWidth ((outWidthR%2) == 1 ? outWidthR + 1 : outWidthR)

#define SHIFT 0


#endif
