'''
É necessário instalar uma versão específica do GLAS para este script rodar. O
código para instalar rodando pip é o seguinte:

    pip install --upgrade git+git://github.com/drcassar/glas@dev3

'''
from pprint import pprint
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

from glas import GlassSearcher as Searcher
from glas.constraint import Constraint
from glas.predict import Predict

from HEAs import burgers, df, T0, poisson, delta_volume, shear


elements = df['elements'].values

def normalizer(alloys):
    soma = np.sum(alloys, axis=1).reshape((-1, 1))
    norm_alloys = alloys / soma
    return norm_alloys


###############################################################################
#                             Classes Tipo Predict                            #
##############################################################################+

class BasePredict(Predict):
    def __init__(self):
        super().__init__()

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool)


class PredictBurgers(BasePredict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        alloys = population_dict['population_array']

        # mudar a partir daqui
        alloys = normalizer(alloys)
        value = burgers(alloys)
        return value


class PredictShear(BasePredict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        alloys = population_dict['population_array']

        # mudar a partir daqui
        alloys = normalizer(alloys)
        value = shear(alloys)
        return value


###############################################################################
#                           Classes Tipo Constraint                           #
##############################################################################+

class ConstraintPoisson(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):

        alloys = population_dict['population_array']

        # mudar a partir daqui
        alloys = normalizer(alloys)
        value = poisson(alloys)
        bad = value <= self.config['min']

        distance_min = self.config['min'] - value
        distance = np.zeros(len(value))
        distance[bad] += distance_min[bad]

        penalty = bad * base_penalty + distance**2
        # fim da mudança

        return penalty


class ConstraintT0(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):

        alloys = population_dict['population_array']

        # mudar a partir daqui
        alloys = normalizer(alloys)

        b = burgers(alloys)
        G = shear(alloys)
        dv = delta_volume(alloys)
        v = poisson(alloys)

        value = T0(alloys, b, G, v, dv)
        bad = value >= self.config['max']

        distance_max = value - self.config['max']
        distance = np.zeros(len(value))
        distance[bad] += distance_max[bad]

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


class ConstraintComplexity(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        num_elements = population_dict['population_array'].astype(bool).sum(axis=1)
        logic1 = num_elements < self.config['min_elements']
        logic2 = num_elements > self.config['max_elements']
        bad = np.logical_or(logic1, logic2)

        distance_min = self.config['min_elements'] - num_elements
        distance_max = num_elements - self.config['max_elements']

        distance = np.zeros(len(num_elements))
        distance[logic1] += distance_min[logic1]
        distance[logic2] += distance_max[logic2]

        penalty = bad * base_penalty + distance**2
        return penalty



###############################################################################
#                            Configuração de Busca                            #
##############################################################################+

design = {
    'burgers': {
        'class': PredictBurgers,
        'name': 'burgers',
        'use_for_optimization': True,
        'config': {
            'min': 2,
            'max': 2.6,
            'objective': 'minimize',
            'weight': 1,
        }
    },

    'shear': {
        'class': PredictShear,
        'name': 'shear',
        'use_for_optimization': True,
        'config': {
            'min': 10,
            'max': 300,
            'objective': 'maximize',
            'weight': 1,
        }
    },

}

constraints = {
    'elements': {
        'class': ConstraintElements,
        'config': {
            'Al': [0.10, 0.35],
        },
    },

    'T0': {
        'class': ConstraintT0,
        'config': {
            'max': 0.15,
        },
    },

    # 'Poisson': {
    #     'class': ConstraintPoisson,
    #     'config': {
    #         'min': 0.45,
    #     },
    # },

    # 'complexity': {
    #     'class': ConstraintComplexity,
    #     'config': {
    #         'min_elements': 4,
    #         'max_elements': 5,
    #     },
    # },

}

config = {
    'num_generations': 2000,
    'population_size': 400,
    'hall_of_fame_size': 1,
    'num_repetitions': 1,
    'compound_list': list(elements),
}


###############################################################################
#                                    Busca                                    #
##############################################################################+

all_hof = []
for _ in range(config['num_repetitions']):
    S = Searcher(config, design, constraints)
    S.start()
    S.run(config['num_generations'])
    all_hof.append(S.hof)


###############################################################################
#                                    Report                                   #
##############################################################################+

print()
print(f'--------  REPORT -------------------')
print(f'--------  Design configuration -------------------')
pprint(config)
print()
pprint(design)
print()
print(f'--------  Constraints -------------------')
pprint(constraints)
print()

for p, hof in enumerate(all_hof):
    print()
    print(f'------- RUN {p+1} -------------')
    print()
    for n, ind in enumerate(hof):
        print(f'Position {n+1} (mol%)')
        S.report_dict(ind, verbose=True)
