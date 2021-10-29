from pprint import pprint
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import math
from scipy import constants
from mendeleev import element
from deap import tools

from glas import GlassSearcher as Searcher
from glas.constraint import Constraint
from glas.predict import Predict

from HEAs import df, burgers, shear, K, poisson, delta_volume, T0, dEb, stress, parVEC, parPhi

elements=df['elements'].values

def normalizer (alloys):
    total=np.sum(alloys,axis=1).reshape((-1,1))
    norm_alloys=alloys/total
    return norm_alloys


###############################################################################
#                             Predict Class                                   #
##############################################################################+

class PredictK(Predict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        alloys=population_dict['population_array']
        alloys=normalizer(alloys)
        G=shear(alloys)
        b=burgers(alloys)
        value=K(G,b)
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
        alloys=population_dict['population_array']
        alloys=normalizer(alloys)
        b=burgers(alloys)
        G=shear(alloys)
        v=poisson(alloys)
        dv=delta_volume(alloys)
        T0_p=T0(alloys,b,G,v,dv)
        dEb_p=dEb(alloys,b,G,v,dv)
        value=stress(T0_p, dEb_p)
        return value 

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool)    
 

###############################################################################
#                           Constraint Class                                  #
##############################################################################+

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
    

###############################################################################
#                            Search Class                                     #
##############################################################################+

class Searcher(Searcher):
    '''Searcher class with some extra configurations

    Changes:
        + Now the class reports the number of repetition that it is running

    '''
    def __init__(self, config, design, constraints={}, run_num=0):

        self.run_num = run_num
        super().__init__(config, design, constraints)

    def callback(self):
        best_fitness = min([ind.fitness.values[0] for ind in self.population])
        print(
            'Finished generation {1}/{0}. '.format(
                str(self.generation).zfill(3),
                str(self.run_num).zfill(2)
            ),
            f'Best fitness is {best_fitness:.3g}. '
        )

        if self.generation % self.report_frequency == 0:
            if best_fitness < self.base_penalty:
                best_ind = tools.selBest(self.population, 1)[0]
                print('\nBest individual in this population (in mol%):')
                self.report_dict(best_ind, verbose=True)


    

###############################################################################
#                         Design, Constraint & Config                         #
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
#                                    Search                                   #
##############################################################################+

all_hof = []
for i in range(config['num_repetitions']):
    S = Searcher(config, design, constraints, i+1)
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

for p, hof in enumerate(all_hof):
    print()
    print(f'------- RUN {p+1} -------------')
    print()
    for n, ind in enumerate(hof):
        print(f'Position {n+1} (mol%)')
        print(f'Fitness: {S.fitness_function([ind])[0]:5f}')
        S.report_dict(ind, verbose=True)
