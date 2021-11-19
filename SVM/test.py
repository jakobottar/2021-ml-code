import numpy as np
import math
X = [[1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12]]
gamma = 1

# xvals = np.zeros((len(X), len(X)))
# for i in range(len(X)):
#     for j in range(len(X)):
#         xvals[j, i] = math.exp(-(np.linalg.norm(X[i] - X[j])**2)/gamma)

# print(xvals)

# x = np.array([[1], [2], [3]])
# y = np.array([4, 5, 6])
y = [[1,2,3],
     [4,5,6]]
x = [[1,2],
     [3,4],
     [5,6]]
b = np.broadcast(y, y)

# out = np.empty(b.shape)
# out.flat = [u+v for (u,v) in b]
# print(b.nd)
for u, v in b:
    print(u, v)
# print(out)