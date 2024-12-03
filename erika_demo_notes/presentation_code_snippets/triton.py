from mylib import program_id, arange, load, store

# fmt: off


BLOCK = 512 
@jit 
def add(X, Y, Z, N): 
    pid = program_id(0) # block of indices 
    idx = pid * BLOCK + arange(BLOCK) 
    mask = idx < N
    x = load(X + idx, mask=mask) 
    y = load(Y + idx, mask=mask)     
    store(Z + idx, x + y, mask=mask) 


# fmt: on
