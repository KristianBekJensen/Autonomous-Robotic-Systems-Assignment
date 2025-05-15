from fitness import Fitness
from maze_solver import ExploreController, TargetController
from trajectory_recorder import load

# Calculate total genome length:
# number of sensors + wheel speeds
input_size = 6 + 2
hidden_size = 10
output_size = 2

genome_length = (
    hidden_size * input_size # W1
    + hidden_size # b1
    + output_size * hidden_size # W2
    + output_size # b2
)
#Find best from population
#best = min(load("navigate_to_goal.pkl"), key=lambda ind: ind.fitness).phenome
best = load("finalphenome.pkl")
#Controllers
close_controller = TargetController( #or None
          genotype=best,
          input_size= 6 + 2 + 1 + 1,
          hidden_size=hidden_size,
          output_size=output_size
      )

controller = ExploreController

# Arguments
num_sensors = 6
start_pop_filename = None
max_steps = 2800
random_map = False
pop_size = 16
generations = 100000

# Define Fitness weights
collision_weight = 0.0
time_weight = 0.0
dist_weight = 0.0
exploration_weight = 1000.0
speed_weight = 0.5 
targets_collected_weight = -500

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
    f"multi_targets_sensors{num_sensors}"
    f"_Msteps{int(max_steps/10)}"
    f"_r{'T' if random_map else 'F'}"
    f"_P{pop_size}"
    f"_c{collision_weight}"
    f"_e{exploration_weight/10}"
    f"_s{speed_weight}"
    f"_t{targets_collected_weight}"
)
