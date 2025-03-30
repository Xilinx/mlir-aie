m = 48
k = 100
n = 56

mtk = 4*k
ktn = 4*k

# mtk = k
# ktn = k

# Specify starting point and step below
# 4 rows for phoenix
M_min = 4 * m
M_step = 2*M_min
M_max = 8000

# 4 columns for phoenix
N_min = 4 * n
N_step = 2*N_min
N_max = 8000

K_min = mtk
K_step = 4*K_min
K_max = 8000

counter = 0
for M in range(M_min, M_max, M_step):
    for K in range(K_min, K_max, K_step):
        for N in range(N_min, N_max, N_step):

            print(f"M={M}, K={K}, N={N}")

            counter +=1

print(counter)




# k = 40


# for K in range(k, 5000, k):

#     for mtk in range(k, K, k):

#         if K % mtk == 0:
#             print(mtk)

#     print(f"K={K}")
#     print()
