from Fitness import Fitness
from maze_solver_nn import ExploreController, TargetController
from trajectory_recorder import load_pop

# Number of sensors 
num_sensors = 6

# MLP dimensions (must match maze_solver.py)
# input = num_sensor + wheel left right
input_size   = 6 + 2 + 1 + 1
hidden_size  = 10
output_size  = 2

#Controllers
close_controller = None

controller = TargetController

# calculate total genome length:
genome_length = (
      hidden_size * input_size    # W1
    + hidden_size                 # b1
    + output_size * hidden_size   # W2
    + output_size                 # b2
)

start_pop_filename = None

max_steps = 2400
random_map = False

pop_size    = 16
generations = 100000

# fitness
fitness = Fitness(collision_weight=0.0,
                time_weight=0.0,
                dist_weight=0.0,
                exploration_weight=1000.0,
                speed_weight=0.5 ,
                targets_collected_weight = -500)
fitness_func = fitness.linar_fitness