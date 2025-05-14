from maze_solver_nn import MazeSolver
from trajectory_recorder import load_pop
from config_nn import num_sensors, input_size, hidden_size, output_size, genome_length, pop_size, generations

filename = "map_explored1000_speed5_col20_16pop_time24_gen_30.pkl"

# Show best perfromance
problem = MazeSolver(
            maximize=False,
            visualization=True,
            num_sensors=num_sensors,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            max_steps=100000,
            random=110
        )
problem.evaluate(max(load_pop(filename)).phenome)