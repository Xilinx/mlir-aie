# Run test
def construct_test(f):
    print("\nTEST:", f.__name__)
    f()


# This is a utility function used in creating tests to pretty print arrays in a way that's easy to see tiles
def array_printer(arr):
    pad = len(str(len(arr) * len(arr[0])))
    print("[")
    for i in range(len(arr)):
        print("\t[", end="")
        for j in range(len(arr[0])):
            print(f"{arr[i][j]:{pad}},", end="")
        print("],")
    print("]")
