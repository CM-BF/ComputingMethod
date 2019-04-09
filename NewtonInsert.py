'''
Describe: 程序 3 （1）（2）
Date: 09 Apr 2019
Author: 归舒睿
'''


import numpy as np
inputfile = open("data/NewtonInsert.txt", "r")
line = inputfile.readline()
data = []
count = 0
while line != '':
    count += 1
    line = line.strip('\n').split(' ')
    data.append({'x': int(line[0]), 'y':int(line[1])})
    line = inputfile.readline()

g = np.zeros((count, count), np.float)

for j in range(count):
    for i in range(j, count):
        if j == 0:
            g[i][j] = data[i]['y']
        else:
            g[i][j] = (g[i][j-1] - g[i-1][j-1]) / (data[i]['x'] - data[i-j]['x'])


predict = [1965, 2012]

def Newton(x):
    global count
    N = 0
    t = 1
    for k in range(count):
        N += g[k][k] * t
        t *= x - data[k]['x']
    return N

print('1965: ', Newton(predict[0]), '\n2012: ', Newton(predict[1]))
