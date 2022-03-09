from nevergrad.benchmark.experiments import yabbob
from nevergrad.benchmark.experiments import yabigbbob
from nevergrad.benchmark.experiments import yasmallbbob
from nevergrad.benchmark.experiments import yahdbbob

file = './Required files/instances_small_10_50.txt'

benchmarks = [
        # yabbob, 
        yasmallbbob, 
        # yabigbbob, 
        # yahdbbob
        ]

names = [
    "hm",
    "rastrigin",
    "griewank",
    "rosenbrock",
    "ackley",
    "lunacek",
    "deceptivemultimodal",
    "bucherastrigin",
    "multipeak",
]
names += ["sphere", "doublelinearslope", "stepdoublelinearslope"]
names += ["cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid", "discus", "bentcigar"]
names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]


for benchmark in benchmarks:
    for name in names:
        functions, functions_configurations, budgets = benchmark(name=name)
        for config in functions_configurations:
            for budget in budgets:
                if budget in [10, 50]:
                    with open(file=file, encoding = 'utf-8', mode='a') as f:
                        f.write(f'{benchmark.__name__} --name={name} --rotation={config.get("rotation")} --d={config.get("block_dimensions")} --budget={budget}\n')