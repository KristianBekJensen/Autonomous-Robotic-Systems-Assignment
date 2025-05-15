from fitness import Fitness
from maze_solver import ExploreController, TargetController
from trajectory_recorder import load

# Calculate total genome length:
# number of sensors + wheel speeds + dist to target + angle to target
input_size = 6 + 2 + 1 + 1
hidden_size = 10
output_size = 2
genome_length = (
    hidden_size * input_size # W1
    + hidden_size # b1
    + output_size * hidden_size # W2
    + output_size # b2
)

#Controllers
close_controller = None

controller = TargetController

# Arguments
num_sensors = 6
start_pop_filename = None
max_steps = 1200
random_map = False
pop_size = 16
generations = 100000

# Define Fitness weights
collision_weight = 0.0
time_weight = 0.0
dist_weight = 2.0
exploration_weight = 0.0
speed_weight = 0.0
targets_collected_weight = 0.0

fitness = Fitness(
    collision_weight=collision_weight,
    time_weight=time_weight,
    dist_weight=dist_weight,
    exploration_weight=exploration_weight,
    speed_weight=speed_weight,
    targets_collected_weight=targets_collected_weight
)
fitness_func = fitness.linear_fitness

# Make Filename
save_as = (
    f"targetcontroller_sensors{num_sensors}"
    f"_Msteps{int(max_steps/10)}"
    f"_r{'T' if random_map else 'F'}"
    f"_P{pop_size}"
    f"_c{collision_weight}"
    f"_e{exploration_weight/10}"
    f"_s{speed_weight}"
    f"_t{targets_collected_weight}"
)