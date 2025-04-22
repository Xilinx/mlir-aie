# max number of rows tiles to submit each BD for A and B at a time
# Each BD represents:
# 1) one row tile for A, i.e., m * K,
# 2) one col tile for B, i.e., n * K,
# 3) one out tile for C, i.e., m * n
max_BDs_per_A_B = 5

# Number of total row tiles
total_row_tiles = 2

# Number of total col tiles
total_col_tiles = 2

# keep track of the row and col indices
row_index = 0
col_index = 0

# counter for initial BD assignment
initial_BD_cnt = 0

should_break = False

# First, submit the initial BDs for each A and B
for i in range(total_row_tiles):

    # if col tiles have finished,
    # start submitting the next row (thus col_index = 0)
    col_index = 0

    for j in range(total_col_tiles):

        print(
            f"BD count: {initial_BD_cnt}, row_index: {row_index}, col_index: {col_index}"
        )

        # break innermost loop when max BDs reached
        # or total tiles reached
        if (initial_BD_cnt == max_BDs_per_A_B - 1) or (
            initial_BD_cnt == total_row_tiles * total_col_tiles - 1
        ):
            should_break = True
            break

        col_index += 1
        initial_BD_cnt += 1

    # break also outermost loop when max BDs reached
    if should_break:
        break

    row_index += 1


# The two loops above will finish either by break or by very low number of row and col tiles.
# In both cases, the row_index and col_index variables store the already processed indices.
# We increase the indices to point to the next row and col tiles.

# Always increase the col index to point to the next tile
col_index += 1

# Increase the row index in case we reached all the col tiles
if col_index == total_col_tiles:
    col_index = 0
    row_index += 1


print("Reconfiguration Below")
# this will be the loop for C, and the A, B reconfiguration
for i in range(total_row_tiles):
    for j in range(total_col_tiles):

        # There are A and B need to be reconfigured when we haven't finished processing the rows
        # exclude the first time only, i.e., i=0, j=0, so we know we can reconfigure
        if (i > 0 or j > 0) and (row_index < total_row_tiles):

            print(f"row_index: {row_index}, col_index: {col_index}")

            col_index += 1
            if col_index == total_col_tiles:
                col_index = 0
                row_index += 1
