import os
import sys
import time

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.util import wrap_curry
from leap_ec.ops import UniformCrossover

from MazeSolver import MazeSolver
from trajectory_recorder import load_pop, save_pop
import pickle


# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: flip this to False to skip training and load the old population
# ───────────────────────────────────────────────────────────────────────────────
run_training = False
pop_file     =  os.path.join("populations", "final_population.pkl")


# ───────────────────────────────────────────────────────────────────────────────
# EA & simulation parameters
# ───────────────────────────────────────────────────────────────────────────────
action_bits   = 2
num_sensors   = 6
wheel_inputs  = 6
angle_inputs  = 2

genome_length = (2 ** (num_sensors + wheel_inputs + angle_inputs)) * action_bits
pop_size      = 8
generations   = 5


# ───────────────────────────────────────────────────────────────────────────────
# Create the problem (no GUI during training)
# ───────────────────────────────────────────────────────────────────────────────
problem = MazeSolver(
    maximize=False,
    visualization=False,
    num_sensors=num_sensors,
    wheel_inputs=wheel_inputs,
    angle_inputs=angle_inputs
)


# ───────────────────────────────────────────────────────────────────────────────
# Helper to checkpoint the population on disk
# ───────────────────────────────────────────────────────────────────────────────
@wrap_curry
def save(pop, filename):
    save_pop(pop, filename)
    return pop


# ───────────────────────────────────────────────────────────────────────────────
# If we're training, run the EA; otherwise just load the saved population
# ───────────────────────────────────────────────────────────────────────────────
if run_training:
    # ── set up live plots ────────────────────────────────────────────────────
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    p1 = probe.FitnessPlotProbe(ax=ax1, xlim=(0, generations))
    p2 = probe.PopulationMetricsPlotProbe(
        metrics=[probe.pairwise_squared_distance_metric],
        xlim=(0, generations),
        title="Population Diversity",
        ax=ax2
    )
    viz_probes = [p1, p2]
    plt.show(block=False)

    # ── run the evolutionary algorithm ───────────────────────────────────────
    final_pop = generational_ea(
        max_generations=generations,
        pop_size=pop_size,
        problem=problem,
        representation=Representation(
            initialize=create_binary_sequence(length=genome_length)
        ),
        pipeline=[
            ops.tournament_selection(k=2),
            ops.clone,
            UniformCrossover(p_swap=0.4),
            mutate_bitflip(expected_num_mutations=100),
            ops.evaluate,                      # runs MazeSolver.evaluate()
            ops.pool(size=pop_size),
            save(filename="current_population.pkl"),
            probe.FitnessStatsCSVProbe(stream=sys.stdout),
            *viz_probes
        ]
    )

    # ── finalize plots ────────────────────────────────────────────────────────
    plt.ioff()
    if os.environ.get(test_env_var, "") != "True":
        plt.show()
    plt.close(fig)

    # ── save the final population ─────────────────────────────────────────────
    save_pop(final_pop, pop_file)
    print(f"Training complete—saved population to {pop_file!r}.")

else:
    # ── just load the existing population ────────────────────────────────────
    if not os.path.exists(pop_file):
        sys.exit(f"ERROR: no saved population at {pop_file!r}")

    with open(pop_file, 'rb') as f:
        final_pop = pickle.load(f)

    print(f"Loaded {len(final_pop)} individuals from {pop_file!r}.")


# ───────────────────────────────────────────────────────────────────────────────
# Save/report the best and then watch it in Pygame
# ───────────────────────────────────────────────────────────────────────────────
best = min(final_pop, key=lambda ind: ind.fitness)
print("Best individual:", best)

# Visualize in action
visual_problem = MazeSolver(
    maximize=False,
    visualization=True,
    num_sensors=num_sensors,
    wheel_inputs=wheel_inputs,
    angle_inputs=angle_inputs
)
visual_problem.evaluate(best.phenome)
