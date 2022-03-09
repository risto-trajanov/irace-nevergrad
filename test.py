import nevergrad as ng
from scipy.optimize import minimize
from nevergrad.functions import ArtificialFunction
from nevergrad.benchmark.xpbase import Experiment
from nevergrad.benchmark import experiments
from nevergrad.optimization.base import ConfiguredOptimizer
from nevergrad.optimization.optimizerlib import *
from nevergrad.optimization.recastlib import ScipyOptimizer

#yabbob --name=hm --rotation=False --d=2 --budget=12800
parameters_function = {
        'name': 'hm',
        'rotation': False,
        'd': 2
    }
    
benchmark = 'yabbob'
method = getattr(experiments, benchmark)
artificial_function = method(options=parameters_function)
optimizer = ScipyOptimizer(method='Nelder-Mead', options={'adaptive':False, 'fatol': 1.6})
experiment = Experiment(function=artificial_function, optimizer=optimizer, budget=800, seed=263162679)
experiment._run_with_error()