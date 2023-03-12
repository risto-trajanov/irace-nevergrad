# irace-nevergrad

A training instance is defined by a benchmark function, a dimension, a rotation
and the budget available to CMA. We also define “blocks” of instances: all
instances within a block are equal except for the benchmark function and there
are as many instances within a block as benchmark functions. We setup irace so
that, within each race, the first elimination test (FirstTest) happens after seeing
5 blocks and subsequent elimination tests (EachTest) happen after each block.
Moreover, configurations are evaluated by irace on blocks in order of increasing
budget first and increasing dimension second, such that we can quickly discard
poor-performing configurations on small budgets and only good configurations
are evaluated on large ones [26]. The performance criterion optimized by irace is
the objective value of the point recommended by CMA after it has exhausted its
budget. Since Nevergrad validates performance according to the mean loss (as
explained later), the elimination test used by irace is set to t-test. Finally, we
set a maximum of 10 000 individual runs of CMA as the termination criterion
of each irace run. By parallelizing each irace run across 4 CPUs, the runtime of
a single run of irace was around 8 hours.

https://arxiv.org/pdf/2209.04412.pdf

Contribution to Facebook's Nevergrad: https://github.com/facebookresearch/nevergrad/compare/main...ristocma
