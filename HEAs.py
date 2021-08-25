import numpy as np
import pandas as pd

alloy1=np.array([0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
alloy2=np.array([0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
alloy3=np.array([0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])
alloys=([0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])

df=pd.read_csv('parameters.csv', delimiter=';')
atomic_radius=np.array(df['atomic_radius'])
burgers_vector=np.array(df['burgers_vector'])
poisson_ratio=np.array(df['poisson_ratio'])
shear_modulus=np.array(df['shear_modulus'])

def burgers (alloys):
    """Burgers Vector calculation.
    
    Args:
      alloys : array 2D
        Each row of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
    
    Return:
      b : array 1D
        The array contains the burgers vector of the alloys present in this study
    """
    b=(alloys*burgers_vector).sum(axis=1)
    return b
b=burgers(alloys)

def shear (alloys):
    """Shear Modulus calculation.
    
    Args:
      alloys : array 2D
        Each arrow of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
    
    Return:
      G : array 1D
        The array contains the shear modulus in GPa of the alloys present in this study 
    """
    G=(alloys*shear_modulus).sum(axis=1)
    return G 
G=shear(alloys)

def K (G,b,beta=0.18):
    """Hall-Petch constant calculation.
    
    Args:
      G : array 1D
        The array contains the shear modulus in GPa of the alloys present in this study
      b : array 1D
        The array contains the burgers vector of the alloys present in this study
      beta : float number
         Constant which value is 0.18 for FCC structures
    Return:
      K : array 1D
        The array contains the Hall-Petch constants in MPa.µm^1/2 of the alloys present in this study
    """
    K=10*beta*G*b**(1/2)
    return K

def poisson (alloys):
    """Poisson's Ratio calculation.
    
    Args:
      alloys : array 2D
        Each row of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
        
    Return:
      v : array 1D
        The array contains the poisson's ratios of the alloys present in this study
    """
    
    v=(alloys*poisson_ratio).sum(axis=1)
    return v
v=poisson(alloys)

def delta_volume (alloys):
    """Calculation of the atomic volume variation.
        
    Args:
      alloys : array 2D
        Each row of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
    
    Return:
      dv : array 2D
        The array contains the variation of the volume of an individual atom in a FCC unit cell in comparison with its volume considering the composition of the alloy  
        Each line of the array represents one alloy and each column represents the variation in volume of each chemical element 
    """
    #edge_length : array 1D - Contains the edge lenght of a CFC unit cell considering that it is formed by a single type of element
    edge_length=4*atomic_radius/2**(1/2)
    
    #V1 : array 1D - Contains the atomic volume of one atom considering a CFC unit cell formed by a single type of element. Each CFC unit cell contains 4 atoms
    V1=edge_length**3/4
    
    #V2 : array 1D - Contains the atomic volume of one atom inside of a CFC unit cell considering the composition of the alloy
    V2=(alloys*V1).sum(axis=1)
    V2=V2.reshape((-1,1))   
    
    dv=V2-V1
    return dv
dv=delta_volume(alloys)

def T0(alloys, b, G, v, dv, alpha=0.123):
    """Calculation of the intrinsic lattice resistance at 0K
    
    Args:
      alloys : array 2D
        Each row of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
      b : array 1D
        The array contains the burgers vector of the alloys present in this study        
      G : array 1D
        The array contains the shear modulus of the alloys present in this study
      v : array 1D
        The array contains the poisson's ratios of the alloys present in this study
      dv : array 2D
        The array contains the variation of the volume of an individual atom in a FCC unit cell in comparison with its volume considering the composition of the alloy  
        Each line of the array represents one alloy and each column represents the variation in volume of each chemical element
      alpha : float number
        Proportionality constant between the dislocation line tension and Gb2    
    
    Return:
      T0 : array 1D
        The array contains the intrinsic lattice resistance at 0K of the alloys present in this study
    """
 
    b=b.reshape((-1,1))
    p=(1+v)/(1-v)
    s=(alloys*dv**2/b**6).sum(axis=1)
    T0=0.051*alpha**(-1/3)*G*p**(4/3)*0.35*s**(2/3)
    return T0
T0=T0(alloys,b,G,v,dv)

def dEb(alloys, b, G, v, dv, alpha=0.123):
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
      alpha : float number
        Proportionality constant between the dislocation line tension and Gb2
        
    Return:
      dEb : array 1D
        The array contains the total activation energy barrier at 0K of the alloys present in this study
    """
    G1=G*10**9
    b1=b/10**10
    p=(1+v)/(1-v)
    b=b.reshape((-1,1))
    s=(alloys*dv**2/b**6).sum(axis=1)
    dEb=0.274*alpha**(1/3)*G1*b1**3*p**(2/3)*5.7*s**(1/3)
    return dEb
dEb=dEb(alloys,b,G,v,dv)

def stress(T0, dEb, ep=10**-3, T=293):
    """Calculation of the yield strength
    
    Args:
      T0 : array 1D
        The array contains the intrinsic lattice resistance at 0K of the alloys present in this study
      dEb : array 1D
        The array contains the total activation energy barrier at 0K of the alloys present in this study
      ep : float number
        Strain rate
      T : float number
        Temperature in Kelvin
    
    Return:
      stress : array 1D
        The array contains the yield strength in MPa of the alloys present in this study
    """
    #k:Boltzmann constant;
    k=1.38e-23
    
    
    test_parameter=T0*(1-(k*T/dEb*np.log(10**4/ep))**(2/3))
    stress=np.where(
        np.greater(test_parameter,T0*0.4),
        3060*test_parameter,
        3060*T0*np.exp(-1/0.51*k*T/dEb*np.log(10**4/ep)))
    return stress
stress=stress(T0,dEb)
