from pprint import pprint
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import math
from scipy import constants
from mendeleev import element


from glas import GlassSearcher as Searcher
from glas.constraint import Constraint
from glas.predict import Predict


###############################################################################
#                                   Searcher                                  #
##############################################################################+

class SearcherModified(Searcher):
    def __init__(self, config, design, constraints={}):
        super().__init__(
            config=config,
            design=design,
            constraints=constraints,
        )

    def report_dict(self, individual, verbose=True):
        report_dict = {}
        ind_array = np.array(individual).reshape(1,-1)

        pop_dict = {
            'population_array': ind_array,
            'population_weight': self.population_to_weight(ind_array),
            'atomic_array': self.population_to_atomic_array(ind_array),
        }

        if verbose:
            pprint(self.ind_to_dict(individual))

        if verbose:
            print()
            print('Predicted properties of this individual:')

        for ID in self.models:
            y_pred = self.models[ID].predict(pop_dict)[0]
            report_dict[ID] = y_pred
            if verbose:
                print(f'{self.design[ID]["name"]} = {y_pred:.3f} '
                      f'{self.design[ID].get("unit", "")}')

        for ID in self.report:
            y_pred = self.report[ID].predict(pop_dict)[0]
            within_domain = self.report[ID].is_within_domain(pop_dict)[0]
            if within_domain:
                report_dict[ID] = y_pred
                if verbose:
                    print(f'{self.design[ID]["name"]} = {y_pred:.3f} '
                        f'{self.design[ID].get("unit", "")}')
            elif verbose:
                print(f'{self.design[ID]["name"]} = Out of domain')
        
        dict_functions = {
           'VEC': parVEC, 
           'phi': parPhi,
           'K': parK,
           'stress' : parStress,
           
                          } 
        
        for ID in constraints:
            if ID == 'complexity' or ID == 'elements':
                pass
            else:
                print(f'{ID} = %.3f' % dict_functions[ID](ind_array)[0])
        
        if verbose:
            print()
            print()

        return report_dict


###############################################################################
#                          Creation of the Dataframes                         #
##############################################################################+


dfProperties = pd.read_excel(r"properties.xlsx", index_col=0)
atomicSize = dfProperties["atomic_size"] * 100
atomicRadius = dfProperties["atomic_radius"]
meltingTemperature = dfProperties["Tm"]
burgersVector = dfProperties["burgers_vector"]
shearModulus = dfProperties["shear_modulus"]
poissonRatio = dfProperties ["poisson_ratio"]
dfHmix = pd.read_excel(r"Hmix.xlsx", index_col=0)


###############################################################################
#                           Internal Variables                                #
##############################################################################+


elements = np.array([
    "Al",
    "Si",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe", 
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "Zr",
    "Nb",
    "Mo",
    "Hf",
    "Ta",
    "W",
    "Sn"
])

allVEC = {element(i).symbol: element(i).nvalence() for i in elements}
arrVEC = np.array(list(allVEC.values()))

allMeltingT = {element(i).symbol: meltingTemperature[i] for i in elements}
arrMeltingT = np.array(list(allMeltingT.values()))

allAtomicSize = {element(i).symbol: atomicSize[i] for i in elements}
arrAtomicSize = np.array(list(allAtomicSize.values()))

allBurgersVector = {element(i).symbol: burgersVector[i] for i in elements}
arrBurgersVector = np.array(list(allBurgersVector.values()))

allShearModulus = {element(i).symbol: shearModulus[i] for i in elements}
arrShearModulus = np.array(list(allShearModulus.values()))

allPoissonRatio = {element(i).symbol: poissonRatio[i] for i in elements}
arrPoissonRatio = np.array(list(allPoissonRatio.values()))

allAtomicRadius = {element(i).symbol: atomicRadius[i] for i in elements}
arrAtomicRadius = np.array(list(allAtomicRadius.values()))

###############################################################################
#                             Functions                                       #
##############################################################################+

def normalizer(compositions):
    arraySum = np.sum(compositions, axis=1)
    normValues = compositions / arraySum[:, None]
    return normValues


def burgers (compositions):
    """Burgers Vector calculation.
    
    Args:
      alloys : array 2D
        Each row of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
    
    Return:
      b : array 1D
        The array contains the burgers vector of the alloys present in this study
    """
    b=(compositions*arrBurgersVector).sum(axis=1)
    return b


def shear (compositions):
    """Shear Modulus calculation.
    
    Args:
      alloys : array 2D
        Each arrow of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
    
    Return:
      G : array 1D
        The array contains the shear modulus in GPa of the alloys present in this study 
    """
    G=(compositions*arrShearModulus).sum(axis=1)
    return G


def parK (G,b,beta=0.18):
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


def poisson (compositions):
    """Poisson's Ratio calculation.
    
    Args:
      alloys : array 2D
        Each row of the array represents one alloy and each column represents one chemical element
        The sum of each row must be 1
        
    Return:
      v : array 1D
        The array contains the poisson's ratios of the alloys present in this study
    """
    
    v=(compositions*arrPoissonRatio).sum(axis=1)
    return v


def delta_volume (compositions):
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
    edge_length=4*arrAtomicRadius/2**(1/2)
    
    #V1 : array 1D - Contains the atomic volume of one atom considering a CFC unit cell formed by a single type of element. Each CFC unit cell contains 4 atoms
    V1=edge_length**3/4
    
    #V2 : array 1D - Contains the atomic volume of one atom inside of a CFC unit cell considering the composition of the alloy
    V2=(compositions*V1).sum(axis=1)
    V2=V2.reshape((-1,1))   
    
    dv=V2-V1
    return dv


def T0(compositions, b, G, v, dv, alpha=0.123):
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
    s=(compositions*dv**2/b**6).sum(axis=1)
    T0=0.051*alpha**(-1/3)*G*p**(4/3)*0.35*s**(2/3)
    return T0


def dEb(compositions, b, G, v, dv, alpha=0.123):
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
    s=(compositions*dv**2/b**6).sum(axis=1)
    dEb=0.274*alpha**(1/3)*G1*b1**3*p**(2/3)*5.7*s**(1/3)
    return dEb


def parStress(T0, dEb, ep=10**-3, T=293):
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


def parVEC(compositions):
    compNorm = normalizer(compositions)
    VEC = compNorm * arrVEC
    VECFinal = np.sum(VEC, axis=1)
    return VECFinal


def parPhi(compositions):
    compNorm = normalizer(compositions)
    SeBCC = Se(compNorm, 0.68)
    SeFCC = Se(compNorm, 0.74)
    SeMean = (abs(SeBCC) + abs(SeFCC)) / 2
    phi = (Smix(compNorm) - Sh(compNorm)) / SeMean
    return phi


###############################################################################
#                             Predict Class                                   #
##############################################################################+

class PredictK(Predict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        compositions=population_dict['population_array']
        compositions=normalizer(compositions)
        G=shear(compositions)
        b=burgers(compositions)
        value=parK(G,b)
        return value 

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool)
    
class Predictstress(Predict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        compositions=population_dict['population_array']
        compositions=normalizer(compositions)
        b=burgers(compositions)
        G=shear(compositions)
        v=poisson(compositions)
        dv=delta_volume(compositions)
        T0_p=T0(compositions,b,G,v,dv)
        dEb_p=dEb(compositions,b,G,v,dv)
        value=parStress(T0_p, dEb_p)
        return value 

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool) 

###############################################################################
#                           Constraint Class                                  #
##############################################################################+

class ConstraintPhi(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parPhi(population_dict['population_array'])
        bad = value <= self.config['min']

        distance_min = self.config['min'] - value
        distance = np.zeros(len(value))
        distance[bad] += distance_min[bad]

        penalty = bad * base_penalty + distance**2
        return penalty
    
        
class ConstraintVEC(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parVEC(population_dict['population_array'])
        bad = value <= self.config['min']

        distance_min = self.config['min'] - value
        distance = np.zeros(len(value))
        distance[bad] += distance_min[bad]

        penalty = bad * base_penalty + distance**2
        return penalty    


class ConstraintElements(Constraint):
    def __init__(self, config, compound_list, **kwargs):
        super().__init__()
        self.config = config
        elemental_domain = {el: [0, 1] for el in compound_list}
        for el in config:
            elemental_domain[el] = config[el]
        self.elemental_domain = elemental_domain

    def compute(self, population_dict, base_penalty):
        norm_pop = normalizer(population_dict['population_array'])
        distance = np.zeros(population_dict['population_array'].shape[0])

        for n, el in enumerate(self.elemental_domain):
            el_atomic_frac = norm_pop[:, n]
            el_domain = self.elemental_domain.get(el, [0, 0])

            logic1 = el_atomic_frac > el_domain[1]
            distance[logic1] += el_atomic_frac[logic1] - el_domain[1]

            logic2 = el_atomic_frac < el_domain[0]
            distance[logic2] += el_domain[0] - el_atomic_frac[logic2]

        logic = distance > 0
        distance[logic] = (100 * distance[logic])**2 + base_penalty
        penalty = distance

        return penalty


###############################################################################
#                           Search Class                                      #
##############################################################################+

design = {
    
    'K': {
        'class': PredictK,
        'name': 'K: Hall-Petch constant',
        'use_for_optimization': True,
        'config': {
            'min': 100,
            'max': 800,
            'objective': 'maximize',
            'weight': 1,
        }
    },
    
    'stress': {
        'class': Predictstress,
        'name': 'Stress',
        'use_for_optimization': True,
        'config': {
            'min': 0,
            'max': 1000,
            'objective': 'maximize',
            'weight': 1,
        }
    }    
}
constraints = {
    
    # 'elements': {
    #     'class': ConstraintElements,
    #     'config': {
    #         'Mg': [0.10, 0.35],
    #     },
    # },
   
    'phi': {
        'class': ConstraintPhi,
        'config': {
            'min': 20,
        },
    },
 
    'VEC': {
        'class': ConstraintVEC,
        'config': {
            'min': 8,
        },
    },

}

config = {
    'num_generations':200,
    'population_size': 100,
    'hall_of_fame_size': 1,
    'num_repetitions': 2,
    'compound_list': list(elements),
}

###############################################################################
#                                    Search                                    #
##############################################################################+

all_hof = []
for _ in range(config['num_repetitions']):
    S = SearcherModified(config, design, constraints)
    S.start()
    S.run(config['num_generations'])
    all_hof.append(S.hof)


###############################################################################
#                                    Report                                   #
##############################################################################+

print()
print('--------  REPORT -------------------')
print()
print('--------  Design Configuration -------------------')
pprint(config)
print()
pprint(design)
print()
print('--------  Constraints -------------------')
pprint(constraints)
print()

df_list = []
df_list2 = []

for p, hof in enumerate(all_hof):

    dfComp = pd.DataFrame(normalizer(hof), columns=list(elements))*100

    df_list.append(dfComp)
    
    dict_functions = {
           'VEC': parVEC, 
           'phi': parPhi,
           'K': parK,
           'stress' : parStress,
                     } 
        
    for ID in constraints:
        if ID == 'complexity' or ID == 'elements':
            pass
        else:
            dfComp = pd.DataFrame(dict_functions[ID](hof), columns=[ID])
            df_list2.append(dfComp)
            
            
    for ID in design:
         dfComp = pd.DataFrame(dict_functions[ID](hof), columns=[ID])
         df_list2.append(dfComp)
         

    print()
    print(f'------- RUN {p+1} -------------')
    print()
    for n, ind in enumerate(hof):
        print(f'Position {n+1} (mol%)')
        S.report_dict(ind, verbose=True)

dfComp = pd.concat(df_list, axis=0)

dfComp = dfComp.reset_index(drop=True)

dfPar = pd.concat(df_list2, axis=0)

for ID in design:
    dfPar2 = dfPar[ID].dropna().reset_index(drop=True)
    dfComp = dfComp.join(dfPar2)
    
for ID in constraints:
    if ID == 'complexity' or ID == 'elements':
        pass
    else:
        dfPar2 = dfPar[ID].dropna().reset_index(drop=True)
        dfComp = dfComp.join(dfPar2)

now = datetime.now()

dfComp.to_excel(f'{now.strftime("%d%m%Y_%H%M%S")}.xlsx')