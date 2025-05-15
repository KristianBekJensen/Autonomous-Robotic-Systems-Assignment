import math
import numpy as np
from numpy import log2, number

## Main contributors: Noah, Kristian

class Fitness():
    
    def __init__(
        self, 
        collision_weight,
        time_weight,
        dist_weight,
        exploration_weight,
        targets_collected_weight,
        speed_weight
    ):
        
        self.collision_weight = collision_weight
        self.time_weight = time_weight
        self.dist_weight = dist_weight
        self.exploration_weight = exploration_weight
        self.targets_collected_weight = targets_collected_weight
        self.speed_weight = speed_weight

    def linear_fitness(
        self, num_collisions,
        num_time_steps,
        dist_to_target,
        map_unexplored,
        speed,
        targets_collected
        ):

        """ compute a weighted linear fitness function """

        return (
            self.collision_weight * num_collisions
            + self.time_weight * num_time_steps
            + self.dist_weight * dist_to_target
            + self.exploration_weight * map_unexplored
            + self.targets_collected_weight * targets_collected
            + self.speed_weight * speed
        )

def distance_to_target(target, position):
    return math.sqrt((target[0] - position[0]) ** 2 + (target[1] - position[1]) ** 2)


def compute_map_exploration(grid_prob: np.ndarray, threshold: float = 1e-7):
    """ returns the fraction of the map that is still not explored """
    deviation = np.abs(grid_prob - 0.5)
    explored = deviation > threshold
    return (1 - np.sum(explored) / grid_prob.size)
