# -*- coding: utf-8 -*-
"""
Created on Fri May 24 2019

@author: Wesley
"""

import numpy as np
import matplotlib.pyplot as plt
from lib import equation
from scipy.optimize import fsolve
import math

plt.close('all')

M = np.loadtxt(open("route_estudo.csv","r"),delimiter=",",skiprows=1,usecols=(3,2))
M[:,0]=M[:,0]*1000

rmin = 6 #Local. ac. pedestres (69 kV)
R = np.array([[200,8],[400,8]])


y_min = M[:,1] + rmin

numlinhasR = np.shape(R)[0]

for i in range(0,numlinhasR):
    indice = np.where(M[:,0]>=R[i,0])[0][0]
    y_min[indice-2:indice+2] = M[indice-2:indice+2,1]+R[i,1]

plt.figure(1) #Repres. relevo, torres e catenária
plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.plot(M[:,0],M[:,1],'b',M[:,0],y_min,'r')

xt1 = 0   #valor aproximado de xt1
xt2 = 470 #valor aproximado de xt2
xt3 = 920
xt4 = 1465
xt5 = 1805
xt6 = 2270
xt7 = 2580
 
H = 20

i_t1 = np.where(M[:,0]>=xt1)[0][0]
i_t2 = np.where(M[:,0]>=xt2)[0][0]
i_t3 = np.where(M[:,0]>=xt3)[0][0]
i_t4 = np.where(M[:,0]>=xt4)[0][0]
i_t5 = np.where(M[:,0]>=xt5)[0][0]
i_t6 = np.where(M[:,0]>=xt6)[0][0]
i_t7 = np.where(M[:,0]>=xt7)[0][0]

xt1 = M[i_t1,0] #xt1 corrigido
xt2 = M[i_t2,0] #xt2 corrigido
xt3 = M[i_t3,0]
xt4 = M[i_t4,0]
xt5 = M[i_t5,0]
xt6 = M[i_t6,0]
xt7 = M[i_t7,0]

yt1 = M[i_t1,1] + H #coordenada do ponto mais alto da T1
yt2 = M[i_t2,1] + H #coordenada do ponto mais alto da T2
yt3 = M[i_t3,1] + H
yt4 = M[i_t4,1] + H
yt5 = M[i_t5,1] + H
yt6 = M[i_t6,1] + H
yt7 = M[i_t7,1] + H

#Representação das torres
T1 = np.array([[xt1,M[i_t1,1]],[xt1,yt1]])
T2 = np.array([[xt2,M[i_t2,1]],[xt2,yt2]])
T3 = np.array([[xt3,M[i_t3,1]],[xt3,yt3]])
T4 = np.array([[xt4,M[i_t4,1]],[xt4,yt4]])
T5 = np.array([[xt5,M[i_t5,1]],[xt5,yt5]])
T6 = np.array([[xt6,M[i_t6,1]],[xt6,yt6]])
T7 = np.array([[xt7,M[i_t7,1]],[xt7,yt7]])

#plotar torre
plt.figure(1)
plt.plot(T1[:,0],T1[:,1],'k')
plt.plot(T2[:,0],T2[:,1],'k')
plt.plot(T3[:,0],T3[:,1],'k')
plt.plot(T4[:,0],T4[:,1],'k')
plt.plot(T5[:,0],T5[:,1],'k')
plt.plot(T6[:,0],T6[:,1],'k')
plt.plot(T7[:,0],T7[:,1],'k')

#Definição dos parâmetros da catenária 
Tnom = 140e3
Tmax = 0.25*Tnom
ms = 1350*9.81/1000
T0 = 0.97*Tmax

######################################################
#Cálculo da catenária 1
param = (xt1, xt2,yt1,yt2,T0,ms)
x0 = fsolve(equation,xt1,args=param)[0]
y0 = yt1 - T0/ms*(math.cosh(ms/T0*(xt1-x0))-1)

catenaria1_x = np.arange(xt1-x0,xt2-x0,1)
catenaria1_y = np.array([T0/ms*(math.cosh(ms/T0*x)-1) for x in catenaria1_x])

catenaria1_x_real = catenaria1_x + x0
catenaria1_y_real = catenaria1_y + y0

plt.figure(1)
plt.plot(catenaria1_x_real,catenaria1_y_real,'m')


#Gráfico da tração 1
T = T0 + catenaria1_y*ms
Vtmax = Tmax*np.ones(len(catenaria1_x_real))

plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria1_x_real,T,'b',catenaria1_x_real,Vtmax,'r')

######################################################
#Cálculo da catenária 2
T01 = 0.97*Tmax
param = (xt2, xt3,yt2,yt3,T01,ms)
x0 = fsolve(equation,xt1,args=param)[0]
y0 = yt2 - T0/ms*(math.cosh(ms/T01*(xt2-x0))-1)

catenaria1_x = np.arange(xt2-x0,xt3-x0,1)
catenaria1_y = np.array([T01/ms*(math.cosh(ms/T01*x)-1) for x in catenaria1_x])

catenaria1_x_real = catenaria1_x + x0
catenaria1_y_real = catenaria1_y + y0

plt.figure(1)
plt.plot(catenaria1_x_real,catenaria1_y_real,'m')


#Gráfico da tração 2
T = T01 + catenaria1_y*ms
Vtmax = Tmax*np.ones(len(catenaria1_x_real))

plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria1_x_real,T,'b',catenaria1_x_real,Vtmax,'r')

######################################################
#Cálculo da catenária 3
T02 = 0.97*Tmax
param = (xt3, xt4,yt3,yt4,T02,ms)
x0 = fsolve(equation,xt1,args=param)[0]
y0 = yt3 - T02/ms*(math.cosh(ms/T02*(xt3-x0))-1)

catenaria1_x = np.arange(xt3-x0,xt4-x0,1)
catenaria1_y = np.array([T02/ms*(math.cosh(ms/T02*x)-1) for x in catenaria1_x])

catenaria1_x_real = catenaria1_x + x0
catenaria1_y_real = catenaria1_y + y0

plt.figure(1)
plt.plot(catenaria1_x_real,catenaria1_y_real,'m')

#Gráfico da tração 3
T = T02 + catenaria1_y*ms
Vtmax = Tmax*np.ones(len(catenaria1_x_real))

plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria1_x_real,T,'b',catenaria1_x_real,Vtmax,'r')

######################################################
#Cálculo da catenária 4
T03 = 0.97*Tmax
param = (xt4, xt5,yt4,yt5,T03,ms)
x0 = fsolve(equation,xt1,args=param)[0]
y0 = yt4 - T03/ms*(math.cosh(ms/T03*(xt4-x0))-1)

catenaria1_x = np.arange(xt4-x0,xt5-x0,1)
catenaria1_y = np.array([T03/ms*(math.cosh(ms/T03*x)-1) for x in catenaria1_x])

catenaria1_x_real = catenaria1_x + x0
catenaria1_y_real = catenaria1_y + y0

plt.figure(1)
plt.plot(catenaria1_x_real,catenaria1_y_real,'m')


#Gráfico da tração 4
T = T03 + catenaria1_y*ms
Vtmax = Tmax*np.ones(len(catenaria1_x_real))

plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria1_x_real,T,'b',catenaria1_x_real,Vtmax,'r')

######################################################
#Cálculo da catenária 5
T04 = 0.99*Tmax
param = (xt5, xt6,yt5,yt6,T04,ms)
x0 = fsolve(equation,xt1,args=param)[0]
y0 = yt5 - T04/ms*(math.cosh(ms/T04*(xt5-x0))-1)

catenaria1_x = np.arange(xt5-x0,xt6-x0,1)
catenaria1_y = np.array([T04/ms*(math.cosh(ms/T04*x)-1) for x in catenaria1_x])

catenaria1_x_real = catenaria1_x + x0
catenaria1_y_real = catenaria1_y + y0

plt.figure(1)
plt.plot(catenaria1_x_real,catenaria1_y_real,'m')


#Gráfico da tração 5
T = T04 + catenaria1_y*ms
Vtmax = Tmax*np.ones(len(catenaria1_x_real))

plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria1_x_real,T,'b',catenaria1_x_real,Vtmax,'r')

######################################################
#Cálculo da catenária 6 e 7
T05 = 0.97*Tmax
param = (xt6, xt7,yt6,yt7,T05,ms)
x0 = fsolve(equation,xt1,args=param)[0]
y0 = yt6 - T05/ms*(math.cosh(ms/T05*(xt6-x0))-1)

catenaria1_x = np.arange(xt6-x0,xt7-x0,1)
catenaria1_y = np.array([T05/ms*(math.cosh(ms/T05*x)-1) for x in catenaria1_x])

catenaria1_x_real = catenaria1_x + x0
catenaria1_y_real = catenaria1_y + y0

plt.figure(1)
plt.plot(catenaria1_x_real,catenaria1_y_real,'m')

#Gráfico da tração 6
T = T05 + catenaria1_y*ms
Vtmax = Tmax*np.ones(len(catenaria1_x_real))

plt.figure(2)
plt.xlabel("x[m]")
plt.ylabel("Tração [N]")
plt.plot(catenaria1_x_real,T,'b',catenaria1_x_real,Vtmax,'r')
######################################################

#Cálculo do comprimento
S = T0/ms*(math.sinh(ms/T0*(xt2-x0))-math.sinh(ms/T0*(xt1-x0)))




















