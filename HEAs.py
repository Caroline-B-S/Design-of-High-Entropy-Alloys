import numpy as np
import pandas as pd
from mendeleev import element
from itertools import combinations
from scipy import constants
import math

df=pd.read_csv('parameters.csv', delimiter=';')
atomic_radius=np.array(df['atomic_radius'])
arrAtomicSize=np.array(df['atomic_size'])*100
burgers_vector=np.array(df['burgers_vector'])
poisson_ratio=np.array(df['poisson_ratio'])
shear_modulus=np.array(df['shear_modulus'])
arrMeltingT = np.array(df['Tm'])
elements=np.array(df['elements'])

dfHmix = pd.read_excel(r"Hmix.xlsx", index_col=0)

def normalizer (alloys):
    """Normalization of the composition of the alloy.
    
    Args:
      alloys : array 2D
        Each row of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
    
    Return:
      norm_alloys : array 1D
        The array contains the normalized composition of the alloy. 
        This guarantee that the sum of each row is 1 
    """
    total=np.sum(alloys,axis=1).reshape((-1,1))
    norm_alloys=alloys/total
    return norm_alloys

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

allVEC = {element(i).symbol: element(i).nvalence() for i in elements}
arrVEC = np.array(list(allVEC.values()))

def parVEC(alloys):
    compNorm = normalizer(alloys)
    VEC = compNorm * arrVEC
    VECFinal = np.sum(VEC, axis=1)
    return VECFinal

def Smix(compNorm):
    x = np.sum(np.nan_to_num((compNorm) * np.log(compNorm)), axis=1)
    Smix = -constants.R * 10 ** -3 * x
    return Smix

def Tm(compNorm):
    Tm = np.sum(compNorm * arrMeltingT, axis=1)
    return Tm

def Hmix(compNorm):
    elements_present = compNorm.sum(axis=0).astype(bool)
    compNorm = compNorm[:, elements_present]
    element_names = elements[elements_present]
    Hmix = np.zeros(compNorm.shape[0])
    for i, j in combinations(range(len(element_names)), 2):
        Hmix = (
            Hmix
            + 4
            * dfHmix[element_names[i]][element_names[j]]
            * compNorm[:, i]
            * compNorm[:, j]
        )   
    return Hmix

def Sh(compNorm):
    Sh = abs(Hmix(compNorm)) / Tm(compNorm)
    return Sh

def csi_i(compNorm, AP):
    supportValue = np.sum((1/6)*math.pi*(arrAtomicSize*2)**3*compNorm, axis=1)
    rho = AP/supportValue
    csi_i = (1/6)*math.pi*rho[:, None]*(arrAtomicSize*2)**3*compNorm
    return csi_i

def deltaij(i, j, newCompNorm, newArrAtomicSize, csi_i_newCompNorm, AP):
    element1Size = newArrAtomicSize[i]*2
    element2Size = newArrAtomicSize[j]*2
    deltaij = ((csi_i_newCompNorm[:,i]*csi_i_newCompNorm[:,j])**(1/2)/AP)*(((element1Size - element2Size)**2)/(element1Size * element2Size))*(newCompNorm[:,i] * newCompNorm[:,j])**(1/2)
    return deltaij

def y1_y2(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    elements_present = compNorm.sum(axis=0).astype(bool)
    newCompNorm = compNorm[:, elements_present]
    newCsi_i_compNorm = csi_i_compNorm[:, elements_present]
    newArrAtomicSize = arrAtomicSize[elements_present]
    y1 = np.zeros(newCompNorm.shape[0])
    y2 = np.zeros(newCompNorm.shape[0])
    for i, j in combinations(range(len(newCompNorm[0])), 2):
        deltaijValue = deltaij(i, j, newCompNorm, newArrAtomicSize, newCsi_i_compNorm, AP)
        y1 += deltaijValue * (newArrAtomicSize[i]*2 + newArrAtomicSize[j]*2) * (newArrAtomicSize[i]*2*newArrAtomicSize[j]*2)**(-1/2)
        y2_ = np.sum((newCsi_i_compNorm/AP) * (((newArrAtomicSize[i]*2*newArrAtomicSize[j]*2)**(1/2)) / (newArrAtomicSize*2)), axis=1)
        y2 += deltaijValue * y2_
    return y1, y2

def y3(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    x = (csi_i_compNorm/AP)**(2/3)*compNorm**(1/3)
    y3 = (np.sum(x, axis=1))**3
    return y3

def Z(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    Z = ((1+AP+AP**2) - 3*AP*(y1Values+y2Values*AP) - AP**3*y3Values) * (1-AP)**(-3)
    return Z

def eq4B(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    eq4B = -(3/2) * (1-y1Values+y2Values+y3Values) + (3*y2Values+2*y3Values) * (1-AP)**-1 + (3/2) * (1-y1Values-y2Values-(1/3)*y3Values) * (1-AP)**-2 + (y3Values-1) * np.log(1-AP)
    return eq4B

def Se(compNorm, AP):
    Se = (eq4B(compNorm, AP) - np.log(Z(compNorm, AP)) - (3-2*AP) * (1-AP)**-2 + 3 + np.log((1+AP+AP**2-AP**3) * (1-AP)**-3)) * constants.R*10**-3
    return Se

def parPhi(alloys):
    compNorm = normalizer(alloys)
    SeBCC = Se(compNorm, 0.68)
    SeFCC = Se(compNorm, 0.74)
    SeMean = (abs(SeBCC) + abs(SeFCC)) / 2
    phi = (Smix(compNorm) - Sh(compNorm)) / SeMean
    return phi

if __name__ == '__main__':

    alloy1=np.array([0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    alloy2=np.array([0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    alloy3=np.array([0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])
    alloys=([0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0])

    b=burgers(alloys)
    G=shear(alloys)
    v=poisson(alloys)
    dv=delta_volume(alloys)
    T0=T0(alloys,b,G,v,dv)
    dEb=dEb(alloys,b,G,v,dv)
    stress=stress(T0,dEb)
