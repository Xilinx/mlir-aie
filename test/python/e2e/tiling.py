from pprint import pprint


def m_n_tiling(rows, cols, m, n, word_size):
    return [
        [(matrix_dims[1] // 4), (matrix_dims[0] // word_size) * 4],
        [(matrix_dims[0] // word_size // 4), 4],
        [4, (matrix_dims[1] // word_size)],
        [4, 1],
    ]


pprint(m_n_tiling(16, 16, word_size=1))

# divide number of columns by n