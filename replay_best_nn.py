import pickle
from maze_solver_nn import MazeSolver
from config_nn import num_sensors, input_size, hidden_size, output_size


# Load the final population
with open("populations/final_nn_pop.pkl", "rb") as f:
    final_pop = pickle.load(f)


# Pick the best individual (min because we are minimizing the fitness)
best = min(final_pop, key=lambda ind: ind.fitness)
print("Best fitness:", best.fitness)


# Visualize the best individual of the population
visual_solver = MazeSolver(
    maximize=False,
    visualization=True,
    num_sensors=num_sensors,
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)

# run the best individual
visual_solver.evaluate(best.phenome)
