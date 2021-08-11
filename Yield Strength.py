import numpy as np


alloys=([0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])

from functions import burgers
from functions import shear
from functions import poisson
from functions import delta_volume

b=burgers(alloys)
G=shear(alloys)
v=poisson(alloys)
dv=delta_volume(alloys)

def T0(alloys, b, G, v, dv):
    """Calculation of the intrinsic lattice resistance at 0K
    
    Args:
      alloys : array 2D
        Each line of the array represents one alloy and each column represents one chemical element
        The sum of each line must be 1
      b : array 1D
        The array contains the burgers vector of the alloys present in this study        
      G : array 1D
        The array contains the shear modulus of the alloys present in this study
      v : array 1D
        The array contains the poisson's ratios of the alloys present in this study
      dv : array 2D
        The array contains the variation of the volume of an individual atom in a FCC unit cell in comparison with its volume considering the composition of the alloy  
        Each line of the array represents one alloy and each column represents the variation in volume of each chemical element
    
    Return:
      T0 : array 1D
        The array contains the intrinsic lattice resistance at 0K of the alloys present in this study
    """
    #alpha:Proportionality constant between the dislocation line tension and Gb2 
    alpha=0.123

    T0=[]
    for n, (a,c,d,f,g) in enumerate(zip(alloys,b,G,v,dv)):
        T0=np.append(T0, 0.051*(alpha**(-1/3.))*G[n]*(((1+v[n])/(1-v[n]))**(4./3))*0.35*((np.sum((alloys[n]*(dv[n]**2))/b[n]**6))**(2./3)))
    return T0
T0s=T0(alloys,b,G,v,dv)

def dEb(alloys, b, G, v, dv):
    """Calculation of the total activation energy barrier at 0K
    
    Args:
      alloys : array 2D
        Each line of the array represents one alloy and each column represents one chemical element
        The sum of each line must be 1
      b : array 1D
        The array contains the burgers vector of the alloys present in this study        
      G : array 1D
        The array contains the shear modulus of the alloys present in this study
      v : array 1D
        The array contains the poisson's ratios of the alloys present in this study
      dv : array 2D
        The array contains the variation of the volume of an individual atom in a FCC unit cell in comparison with its volume considering the composition of the alloy  
        Each line of the array represents one alloy and each column represents the variation in volume of each chemical element
    
    Return:
      dEb : array 1D
        The array contains the total activation energy barrier at 0K of the alloys present in this study
    """
    #alpha:Proportionality constant between the dislocation line tension and Gb2 
    alpha=0.123

    dEb=[]
    for n, (a,c,d,f,g) in enumerate(zip(alloys,b,G,v,dv)):
        dEb=np.append(dEb, 0.274*(alpha**(1./3))*(G[n]*10**9)*((b[n]/10**10)**3)*(((1+v[n])/(1-v[n]))**(2./3))*5.70*((np.sum(alloys[n]*(dv[n]**2))/b[n]**6)**(1./3)))
    return dEb
dEbs=dEb(alloys,b,G,v,dv)

def stress(T0s, dEbs):
    """Calculation of the yield strength
    
    Args:
      T0s : array 1D
        The array contains the intrinsic lattice resistance at 0K of the alloys present in this study
      dEbs : array 1D
        The array contains the total activation energy barrier at 0K of the alloys present in this study
    
    Return:
      stress : array 1D
        The array contains the yield strength of the alloys present in this study
    """
    #ep:strain rate; k:Boltzmann constant; T: temperature
    ep=10**-3
    k=1.38e-23
    T=293
    
    stress=[]
    for n, (a,c) in enumerate(zip(T0s, dEbs)):  
        #For high stresses/low temperature regime 
        if T0s[n]*(1-(((k*T/dEbs[n])*(np.log(10**4/ep)))**(2./3)))>0.4*T0s[n]:
            stress=np.append(stress, 3060*T0s[n]*(1-(((k*T/dEbs[n])*(np.log(10**4/ep)))**(2./3))))
        #For low stresses/high temperature regime
    else:
        stress=np.append(stress, 3060*T0s[n]*np.exp((-1/0.51)*(k*T/dEbs[n])*(np.log(10**4/ep))))
    return stress

stressp=stress(T0s,dEbs)
stressf=stressp[:-1]
print((stressf),"MPa")