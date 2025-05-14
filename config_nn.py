# Number of sensors 
num_sensors = 6

# MLP dimensions (must match maze_solver.py)
input_size   = num_sensors + 2 #+ 1 + 1
hidden_size  = 10
output_size  = 2

# calculate total genome length:
genome_length = (
      hidden_size * input_size    # W1
    + hidden_size                 # b1
    + output_size * hidden_size   # W2
    + output_size                 # b2
)

max_steps = 2400
random_map = False

pop_size    = 16
generations = 100000

