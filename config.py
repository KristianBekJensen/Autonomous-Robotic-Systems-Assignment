from fitness import Fitness
from maze_solver import ExploreController, TargetController
from trajectory_recorder import load

# Calculate total genome length:
input_size   = 6 + 2
hidden_size  = 10
output_size  = 2
genome_length = (
      hidden_size * input_size    # W1
    + hidden_size                 # b1
    + output_size * hidden_size   # W2
    + output_size                 # b2
)

#Controllers
close_controller = TargetController( #or None
          genotype=min(load("populations/targetcontoller_sensors6_Msteps120.0_rF_P16_c0.0_e0.0_s0.0_t0.0_gen10.pkl")).phenome,
          input_size= 6 + 2 + 1 + 1,
          hidden_size=hidden_size,
          output_size=output_size
      )

controller = ExploreController

# Aguments
num_sensors = 6
start_pop_filename = None
max_steps   = 2800
random_map  = False
pop_size    = 16
generations = 100000

# Define Fitness weights
collision_weight=0.0
time_weight=0.0
dist_weight=0.0
exploration_weight=1000.0
speed_weight=0.5 
targets_collected_weight=-500

fitness = Fitness(collision_weight=collision_weight,
                time_weight=time_weight,
                dist_weight=dist_weight,
                exploration_weight=exploration_weight,
                speed_weight=speed_weight,
                targets_collected_weight=targets_collected_weight)
fitness_func = fitness.linear_fitness

# Make Filename 
save_as = "multi_targets_" + "sensors" + str(num_sensors) + "_Msteps"+ str(max_steps/10) + "_r" + ("T" if random_map else "F") + "_P" + str(pop_size) + "_c" + str(collision_weight) + "_e" + str(exploration_weight/10) + "_s" + str(speed_weight) + "_t" + str(targets_collected_weight)