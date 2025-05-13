import math
import numpy as np

def fitness(num_collisions: int,
            num_time_steps: int,
            dist_to_target: float,
            map_unexplored: float = 0.0,
            collision_weight: float = 1.0,
            time_weight: float = 1.0,
            dist_weight: float = 20.0,
            exploration_weight: float = 1.0) -> float:

    return (collision_weight * num_collisions
            + time_weight * num_time_steps
            + dist_weight * dist_to_target
            + exploration_weight * map_unexplored)

def distance_to_target(position: tuple[float, float], target: tuple[float, float]) -> float:
    dx = target[0] - position[0]                                                                                                
    dy = target[1] - position[1]
    return math.hypot(dx, dy)

def compute_map_exploration(grid_prob: np.ndarray, threshold: float = 0.3):
    
    deviation = np.abs(grid_prob - 0.5)
    explored = deviation > threshold
    return (1 - np.sum(explored) / explored.size)