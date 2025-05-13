import math
import numpy as np

def fitness(num_collisions: int,
            num_time_steps: int,
            dist_to_target: float,
            max_collisions: int = 100,
            max_time_steps: int = 5000,
            max_dist_to_target: float = 565,
            map_unexplored: float = 0.0,
            collision_weight: float = 1.0,
            time_weight: float = 1.0,
            dist_weight: float = 20.0,
            exploration_weight: float = 10.0) -> float:
    
    fitness_score = (collision_weight * (num_collisions / max_collisions)
            + time_weight * (num_time_steps / max_time_steps)
            + dist_weight * (dist_to_target / max_dist_to_target)
            + exploration_weight * map_unexplored)
    print("\nCalculating fitness -> ")
    print(f"collision factor = {collision_weight} * {num_collisions / max_collisions} = {collision_weight * (num_collisions / max_collisions)}")
    print(f"time factor = {time_weight} * {num_time_steps / max_time_steps} = {time_weight * num_time_steps / max_time_steps}")
    print(f"distance factor = {dist_weight} * {dist_to_target / max_dist_to_target} = {dist_weight * dist_to_target / max_dist_to_target}")
    print(f"exploration factor = {exploration_weight} * {map_unexplored} = {exploration_weight * map_unexplored}")
    print(f"fitness={fitness_score}")
    print()
    return fitness_score

def distance_to_target(position: tuple[float, float], target: tuple[float, float]) -> float:
    dx = target[0] - position[0]                                                                                                
    dy = target[1] - position[1]
    return math.hypot(dx, dy)

def compute_map_exploration(grid_prob: np.ndarray, threshold: float = 1e-7):
    deviation = np.abs(grid_prob - 0.5)
    explored = deviation > threshold
    return np.sum(explored) / grid_prob.size