import numpy as np

n = 1
a = 1
b = 2
h = b - 1
e = 10**-4
R = np.zeros((10, 10), np.float)
def f(x):
    return x**4

R[1][1] = (f(1) + f(2))*h/2

for k in range(2, 10):
    sum = 0
    for i in range(1, 2**(k-2) + 1):
        sum += f(a + (2*i - 1)*h/2**(k-1))
    R[k][1] = (R[k-1][1] + h/2**(k-2)*sum)/2
    for j in range(2, k + 1):
        R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1])/(4**(j-1) - 1)
    if abs(R[k][k] - R[k-1][k-1]) < e:
        print(k)
        break

print(R)