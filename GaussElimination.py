import numpy as np
import copy

inputfile = open("data/Gauss.txt", "r")
n = int(inputfile.readline().strip("\n"))
A = np.zeros((n, n), np.float)

for i in range(n):
    line = inputfile.readline().strip("\n").split(' ')
    line = [int(x) for x in line]
    A[i] = line

print('true det: ', np.linalg.det(A))


for i in range(n):
    k = i
    for j in range(i+1, n):
        if abs(A[k][i]) < abs(A[j][i]):
            k = j

    # exchange
    t = copy.deepcopy(A[k])
    A[k] = copy.deepcopy(A[i])
    A[i] = t

    for j in range(i+1, n):
        A[j][i] = A[j][i]/A[i][i]
        A[j][i+1:n] = A[j][i+1:n] - A[j][i] * A[i][i+1:n]
        A[j][i] = 0

det = 1
for i in range(n):
    det *= A[i][i]
print('det after Gauss: ', det)
