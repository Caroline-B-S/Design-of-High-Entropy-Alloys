import numpy as np 

#initial parameters obtained from previous studies 
elements=np.array(["Al", "Co", "Cr", "Cu", "Fe", "Ga", "Ge", "Hf", "In", "Mn", "Mo", "Nb", "Ni", "Sb", "Si", "Sn", "Ta", "Ti", "V", "W", "Zn", "Zr"])
atomic_radius=np.array([1.3126249, 1.2530841, 1.2843456, 1.2744841, 1.2885500, 1.3302253, 1.3169478, 1.6470706, 1.6220227, 1.3550076, 1.4125166, 1.4846432, 1.2460109, 1.5361818, 1.2256518, 1.5748822, 1.4879716, 1.3953989, 1.3054081, 1.4023728, 1.3406209, 1.5539720])
lattice_parameter=np.array([3.7126638, 3.5442570, 3.6326778, 3.6047854, 3.6445697, 3.7624454, 3.7248908, 4.6586192, 4.5877729, 3.8325402, 3.9952002, 4.1992050, 3.5240175, 4.3449782, 3.4666667, 4.4544396, 4.2086193, 3.9467842, 3.6922517, 3.9665093, 3.7918486, 4.3952966])
burgers_vector=np.array([2.6252497, 2.5061682, 2.5686911, 2.5489682, 2.5770999, 2.6604507, 2.6338956, 3.2941413, 3.2440453, 2.7100152, 2.8250332, 2.9692863, 2.4918566, 3.0723635, 2.4513035, 3.1497644, 2.9759432, 2.7907979, 2.6108162, 2.8047456, 2.6812419, 3.1079441])
shear_modulus=np.array([23.3714018, 97.4468766, 69.2044881, 46.3605043, 70.3810701, 9.7842847, 68.0865562, 46.9340368, -184.5161906, 13.7703136, 73.5666217, 40.8107919, 76.0000000, -160.5426745, 67.5588989, -69.7844916, 65.5877827, 30.7263339, 65.6958567, 99.3633824, 34.0999211, -37.0734979])
young_modulus=np.array([64.5582347, 269.1750530, 191.1618146, 128.0604534, 194.4118575, 27.0268833, 188.0737795, 129.6447077, -509.6844262, 38.0373904, 203.2112262, 112.7306227, 209.9328859, -443.4629865, 186.6162450, -192.7639436, 181.1714804, 84.8745783, 181.4700104, 274.4689688, 94.1933533, -102.4071895])


def burgers (alloys):
    """Burgers Vector calculation.
    
    Args:
      alloys : array 2D
        Each line of the array represents one alloy and each column represents one chemical element
        The sum of each line must be 1
    
    Return:
      b : array 1D
        The array contains the burgers vector of the alloys present in this study
    """
    b=[]
    for n in alloys:
        b=np.append(b,sum(n*burgers_vector))
    return b

def shear (alloys):
    """Shear Modulus calculation.
    
    Args:
      alloys : array 2D
        Each line of the array represents one alloy and each column represents one chemical element
        The sum of each line must be 1
    
    Return:
      G : array 1D
        The array contains the shear modulus in GPa of the alloys present in this study 
    """
    G=[]
    for n in alloys:
        G=np.append(G,sum(n*shear_modulus))
    return G 

def poisson (alloys):
    """Poisson's Ratio calculation.
    
    Args:
      alloys : array 2D
        Each line of the array represents one alloy and each column represents one chemical element
        The sum of each line must be 1
        
    Return:
      v : array 1D
        The array contains the poisson's ratios of the alloys present in this study
    """
    v=[]
    for n in alloys:
        v=np.append(v,sum((((young_modulus/(2*shear_modulus))-1)*n)))
    return v

def delta_volume (alloys):
    """Calculation of the atomic volume variation.
        
    Args:
      alloys : array 2D
        Each line of the array represents one alloy and each column represents one chemical element
        The sum of each line must be 1
    
    Return:
      dv : array 2D
        The array contains the variation of the volume of an individual atom in a FCC unit cell in comparison with its volume considering the composition of the alloy  
        Each line of the array represents one alloy and each column represents the variation in volume of each chemical element 
    """
    #edge_length : array 1D - Contains the edge lenght of a CFC unit cell considering a single element
    edge_length=np.array((atomic_radius)*4/np.sqrt(2))
    
    #V1 : array 1D - Contains the atomic volume of a single element considering a CFC unit cell. Each CFC unit cell contains 4 atoms
    V1=np.power(edge_length,3)/4
    
    #V2 : array 1D - Contains the atomic volume of one atom inside of a CFC unit cell considering the composition of the alloy
    V2=[]
    for n in alloys:
        V2=np.append(V2,sum(n*V1))
        
    dv=[V1-V2[0]]
    for n in V2:
        dv=np.append(dv, [V1-n], axis=0)
    dv=np.delete(dv, 0, 0)
    return dv



# def edge_length (atomic_radius):
# """Edge length calculation.

# Args:
# atomic_radius : array 1D
# The array contains the atomic radius of the elements present in this study, considering its arrangement in a CFC unit cell

# Return:
# edge_length : array 1D
# The array contains the edge lenght of a CFC unit cell considering a single element
# """
# edge_length=np.array((atomic_radius)*4/np.sqrt(2))
# return edge_length


# def atomic_volume1 (edge_length):
#     """Atomic volume calculation.
    
#     Args:
#       edge_lenght : array 1D
#         The array contains the edge lenght of a CFC unit cell considering a single element

#     Return:
#       V1 : array 1D  
#         The array contains the atomic volume of a single element considering a CFC unit cell
#         Each CFC unit cell contains 4 atoms 
#     """
#     V1=np.power(edge_length(atomic_radius),3)/4
#     return V1

# def atomic_volume2 (atomic_volume1, alloys):
#     """Atomic volume calculation.
    
#     Args:
#       atomic_volume1 : array 1D
#         The array contains the atomic volume of a single element considering a CFC unit cell
#         Each CFC unit cell contains 4 atoms
#       alloys : array 2D
#         Each line of the array represents one alloy and each column represents one chemical element
#         The sum of each line must be 1
    
#     Return:
#       V2 : array 1D  
#         The array contains the atomic volume of one atom inside of a CFC unit cell considering the composition of the alloy  
#     """
#     V1=atomic_volume1
#     V2=[]
#     for n in alloys:
#         V2=np.append(V2,sum(n*V1))
#     return V2

# def delta_volume (alloys):
#     """Calculation of the atomic volume variation.
        
#     Args:
#       alloys : array 2D
#         Each line of the array represents one alloy and each column represents one chemical element
#         The sum of each line must be 1
    
#     Return:
#       dv : array 2D
#         The array contains the variation of the volume of an individual atom in a FCC unit cell in comparison with its volume considering the composition of the alloy  
#         Each line of the array represents one alloy and each column represents the variation in volume of each chemical element 
#     """
#     V1=atomic_volume1
#     V2=atomic_volume2(atomic_volume1, alloys)
#     dv=[V1-V2[0]]
#     for n in V2:
#         dv=np.append(dv, [V1-n], axis=0)
#     dv=np.delete(dv, 0, 0)
#     return dv

 
        