import numpy as np

from functions import shear
from functions import burgers 
 
alloys=([0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.3, 0.1, 0.3, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])

def K (shear,burgers):
    """Hall-Petch constant calculation.
    
    Args:
      shear : array 1D
        The array contains the shear modulus of the alloys present in this study
      burgers : array 1D
        The array contains the burgers vector of the alloys present in this study
    
    Return:
      K : array 1D
        The array contains the Hall-Petch constants of the alloys present in this study
    """
    K=[]
    beta = 0.18
    G = shear(alloys)
    b = burgers(alloys)
    K=np.append(K,(10*beta*G*b**(1/2)))
    return(K)

print(K(shear,burgers),"MPa.µm^1/2")