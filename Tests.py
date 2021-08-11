import numpy as np

from functions import delta_volume
from functions import poisson
from functions import shear
from functions import burgers

alloys=([0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])
dv=delta_volume(alloys)
v=poisson(alloys)
G=shear(alloys)
b=burgers(alloys)

T0=[]
for n, (a,c,d,f,g) in enumerate(zip(G,v,alloys,dv,b)):
    T0=np.append(T0, 0.051*(0.123**(-1/3.))*G[n]*(((1+v[n])/(1-v[n]))**(4./3))*0.35*((np.sum((alloys[n]*(dv[n]**2))/b[n]**6))**(2./3)))
print(T0)

dEb=[]
for n, (a,c,d,f,g) in enumerate(zip(G, b, v, alloys, dv)):
    dEb=np.append(dEb, 0.274*(0.123**(1./3))*(G[n]*10**9)*((b[n]/10**10)**3)*(((1+v[n])/(1-v[n]))**(2./3))*5.70*((np.sum(alloys[n]*(dv[n]**2))/b[n]**6)**(1./3)))
print(dEb)

ep=10**-3
k=1.38e-23
T=293
stress=[]
#high stresses/low temperature   
for n, (a,b) in enumerate(zip(T0, dEb)):  
    if T0[n]*(1-(((k*T/dEb[n])*(np.log(10**4/ep)))**(2./3)))>0.4*T0[n]:
        stress=np.append(stress, 3060*T0[n]*(1-(((k*T/dEb[n])*(np.log(10**4/ep)))**(2./3))))
#low stresses/high temperature
    else:
        stress=np.append(stress, 3060*T0[n]*np.exp((-1/0.51)*(k*T/dEb[n])*(np.log(10**4/ep))))

print(stress)