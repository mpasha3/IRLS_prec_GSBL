import numpy as np
import cupy as cp

x = np.ones(5)
print(x.shape)
print(type(x))
x = cp.asarray(x)
print(type(x))
x = cp.asnumpy(x)
print(type(x))
print(x.shape)

