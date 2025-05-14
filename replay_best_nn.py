import pickle
from maze_solver_nn import MazeSolver
from config_nn_target_controller import num_sensors, input_size, hidden_size, output_size, genome_length, pop_size, generations, max_steps, random_map, close_controller, start_pop_filename, controller, fitness_func


# Load the final population
with open("navigate_to_goal.pkl", "rb") as f:
    final_pop = pickle.load(f)


# Pick the best individual (min because we are minimizing the fitness)
best = min(final_pop, key=lambda ind: ind.fitness)
print("Best fitness:", best.fitness)

mapSeed = 44

# Visualize the best individual of the population
problem = MazeSolver(
            maximize=False,
            visualization=True,
            num_sensors=num_sensors,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            max_steps=max_steps,
            close_controller=close_controller,
            controller_type=controller,
            fitness_func=fitness_func,
            random = mapSeed
        )

# run the best individual
problem.evaluate(best.phenome)
