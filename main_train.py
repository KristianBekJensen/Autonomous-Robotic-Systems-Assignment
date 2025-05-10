"""An example of an evolutionary algorithm that makes use of LEAP's binary
representation and visualization probes while solving problems in the
broader OneMax family.
"""
from fileinput import filename
import os
import sys

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from MazeSolver import MazeSolver

from trajectory_recorder import load_pop, save_pop
##############################
# main
##############################
action_bits = 2
num_sensors = 6
wheel_inputs = 6
angle_inputs = 2
l = (2**(num_sensors + wheel_inputs + angle_inputs)) * action_bits
pop_size = 5
generations = 5

problem = MazeSolver(False, False, num_sensors, wheel_inputs, angle_inputs)


#############################
# Visualizations
#############################
# Setting up some visualization probes in advance
# Doing it here allow us to use subplots to arrange them nicely
plt.figure(figsize=(15, 5))
plt.subplot(121)
p1 = probe.FitnessPlotProbe(ax=plt.gca(), xlim=(0, generations))
plt.subplot(122)
p2 = probe.PopulationMetricsPlotProbe(
        metrics=[ probe.pairwise_squared_distance_metric ],
        xlim=(0, generations),
        title='Population Diversity',
        ax=plt.gca())
plt.tight_layout()
viz_probes = [p1, p2]

##############################
# Run!
##############################
final_pop = generational_ea(max_generations=generations,pop_size=pop_size,
                         problem=problem,  # Fitness function
                         # Representation
                         representation=Representation(
                             # Initialize a population of integer-vector genomes
                             initialize=create_binary_sequence(length=l)
                         ),
                         # Operator pipeline
                         pipeline=[
                             ops.tournament_selection(k=2),
                             ops.clone,
                             # Apply binomial mutation: this is a lot like
                             # additive Gaussian mutation, but adds an integer
                             # value to each gene
                             mutate_bitflip(expected_num_mutations=500),
                             ops.evaluate,
                             ops.pool(size=pop_size),
                             # Collect fitness statistics to stdout
                             probe.FitnessStatsCSVProbe(stream=sys.stdout),
                             *viz_probes  # Inserting the additional probes we defined above
                         ]
                    )
# If we're not in test-harness mode, block until the user closes the app
if os.environ.get(test_env_var, False) != 'True':
    plt.show()

plt.close('all')

filename = "final_poppulation.pkl"
save_pop(filename, final_pop)
print("Best Individual in final pop: ", max(load_pop(filename)))