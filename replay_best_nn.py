import pickle
from maze_solver import MazeSolver

# ────────────────
# 1) Load the final population
# ────────────────
with open("populations/final_nn_pop.pkl", "rb") as f:
    final_pop = pickle.load(f)

# ────────────────
# 2) Pick the best individual
# ────────────────
# (we’re minimizing, so use min())
best = min(final_pop, key=lambda ind: ind.fitness)
print("Best fitness:", best.fitness)

# ────────────────
# 3) Replay under Pygame
# ────────────────
visual_solver = MazeSolver(
    maximize=False,
    visualization=True,
    num_sensors=6
)

# This call will open a Pygame window and run the entire episode.
visual_solver.evaluate(best.phenome)
