# evolution_nn.py

from json import load
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.util import wrap_curry
from leap_ec.ops import UniformCrossover, tournament_selection, clone

from maze_solver_nn import MazeSolver
from trajectory_recorder import save_pop, load_pop


# ───────────────────────────
# EA & simulation parameters
# ───────────────────────────
num_sensors = 6

# MLP dimensions (must match maze_solver.py)
input_size   = num_sensors + 2 + 1
hidden_size  = 10
output_size  = 2

# calculate total genome length:
genome_length = (
      hidden_size * input_size    # W1
    + hidden_size                 # b1
    + output_size * hidden_size   # W2
    + output_size                 # b2
)

pop_size    = 5
generations = 5


# ───────────────────────────
# Create the problem
# ───────────────────────────
problem = MazeSolver(
    maximize=False,
    visualization=False,
    num_sensors=num_sensors
)
# if true load (sample) from previous population ; if false, random
load_old_pop = False

# ───────────────────────────
# Initializer: Load from previous population
# ───────────────────────────
def init_genome_from_file():
    """
    Load genomes from previous final population file.
    If fewer than pop_size individuals are present, sample with replacement.
    """
    filename = "final_nn_pop.pkl"
    prev_pop = load_pop(filename)

    print(prev_pop)

    if len(prev_pop) < pop_size:
        chosen = np.random.choice(prev_pop, size=pop_size, replace=True)
    else:
        chosen = np.random.choice(prev_pop, size=pop_size, replace=False)

    # Return one genome at a time (LEAP will call this repeatedly)
    for ind in chosen:
        yield ind.genome


# Create an iterator of genomes
genome_iterator = init_genome_from_file()

def init_genome():
    if load_old_pop:
        return next(genome_iterator)
    else:
        # Return one genome at a time (LEAP will call this repeatedly)
        return np.random.uniform(-1.0, 1.0, size=genome_length)


# ───────────────────────────
# A float‐vector initializer
# ───────────────────────────
""" def init_genome():
    
    Return *one* genome (a NumPy array of length genome_length),
    uniformly sampled from [-1,1].  LEAP will call this pop_size times.
    
    return np.random.uniform(-1.0, 1.0, size=genome_length) 
"""

# ───────────────────────────
# A Gaussian‐mutation operator
# ───────────────────────────

@wrap_curry
def mutate_gaussian(pop, sigma: float = 0.1, frac_genes: float = 0.05):
    """
    Generator‐style operator that Gaussian‐mutates an Individual's genome.
    - pop is a generator of Individual objects.
    - sigma is the stddev of N(0,σ²).
    - frac_genes is the fraction of genes to mutate per individual.
    """
    import itertools

    # Peek at the first individual to get genome length
    pop1, pop2 = itertools.tee(pop, 2)
    first = next(pop1, None)
    if first is None:
        # empty population: just return the second stream
        yield from pop2
        return

    n = first.genome.shape[0]
    k = max(1, int(frac_genes * n))

    # Now actually mutate each individual from pop2
    for ind in pop2:
        # clone the entire Individual (including genome)
        child = ind.clone()

        # pick k distinct gene indices
        idxs = np.random.choice(n, size=k, replace=False)
        # add Gaussian noise
        child.genome[idxs] += np.random.normal(0, sigma, size=k)

        # invalidate any old fitness
        child.fitness = None

        yield child


# ───────────────────────────
# Checkpoint helper
# ───────────────────────────
@wrap_curry
def save(pop, filename):
    save_pop(pop, filename)
    return pop


# ───────────────────────────
# Live plotting setup
# ───────────────────────────
""" plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
p1 = probe.FitnessPlotProbe(ax=ax1, xlim=(0, generations))
p2 = probe.PopulationMetricsPlotProbe(
    metrics=[probe.pairwise_squared_distance_metric],
    xlim=(0, generations),
    title="Population Diversity",
    ax=ax2
)
viz_probes = [p1, p2]
plt.show(block=False) """


# ───────────────────────────
# Run the EA
# ───────────────────────────
final_pop = generational_ea(
    max_generations=generations,
    pop_size=pop_size,
    problem=problem,
    representation=Representation(
        # each call returns ONE genome of length genome_length
        initialize=init_genome
    ),
    pipeline=[
        tournament_selection(k=2),
        clone,
        # Uniform crossover works on real arrays
        UniformCrossover(p_swap=0.7),
        # Gaussian mutation
        mutate_gaussian(sigma=0.2, frac_genes=0.3),
        ops.evaluate,               # calls MazeSolver.evaluate()
        ops.pool(size=pop_size),    # keep best pop_size
        save(filename="current_nn_pop.pkl"),
        probe.FitnessStatsCSVProbe(stream=sys.stdout),
        #*viz_probes
    ]
)

# ───────────────────────────
# Finalize plots
# ───────────────────────────
""" plt.ioff()
if os.environ.get(test_env_var, "") != "":
    plt.show()
plt.close(fig) """


# ───────────────────────────
# Save & report
# ───────────────────────────
os.makedirs("populations", exist_ok=True)

final_fname = "final_nn_pop.pkl"
save_pop(final_pop, final_fname)

best = min(final_pop, key=lambda ind: ind.fitness)
print("Training complete.")
print(f"  Best fitness: {best.fitness:.3f}")
print(f"  Saved population to populations/{final_fname!r}")

