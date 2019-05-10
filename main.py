# -*- coding: utf-8 -*-

a = 98
b = 90
c = a - b


print("hello world")
print("value C is ", c)


import numpy as np
import matplotlib.pyplot as plt

M = np.loadtxt(open("route.csv", "r"), delimiter = ",", skiprows = 1, usecols = (3, 2))

M[:, 0] = M[:, 0] * 1000

rmin = 6 #loca ac. pedestres 69kV
R = np.array([[200, 8], [400, 8]])


plt.figure(1)
plt.xlabel("x [m]")
plt.xlabel("y [m]")

y_min = M[:, 1] + rmin

numlinhasR = np.shape(R)[0]

for i in range(0, numlinhasR):
    indice = np.where(M[:, 0] >= R[i, 0])[0][0]
    y_min[indice-2:indice+2] = M[indice-2:indice+2, 1] + R[i, 1]


plt.plot(M[:, 0], M[:, 1], 'b', M[:, 0], y_min, 'r')
