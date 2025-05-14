import pickle
import config_nn_target_controller as conf
from maze_solver_nn import MazeSolver

# load the final saved population
with open("navigate_to_goal.pkl", "rb") as f:
    final_pop = pickle.load(f)


# find the best robot in that population (min because we are minimizing the fitness)
best = min(final_pop, key=lambda ind: ind.fitness)
print("Best fitness:", best.fitness)

mapSeed = 44

# visualize the best robot in the map
problem = MazeSolver(
    maximize=False,
    visualization=True,
    num_sensors=conf.num_sensors,
    input_size=conf.input_size,
    hidden_size=conf.hidden_size,
    output_size=conf.output_size,
    max_steps=conf.max_steps,
    close_controller=conf.close_controller,
    controller_type=conf.controller,
    fitness_func=conf.fitness_func,
    random = mapSeed
)

# run the best individual
problem.evaluate(best.phenome)
