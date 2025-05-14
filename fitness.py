import math
import numpy as np
from numpy import log2, number
class Fitness():
    def __init__(self, 
            collision_weight: float,
            time_weight: float,
            dist_weight: float,
            exploration_weight: float,
            targets_collected_weight:float,
            speed_weight: float):
        
        self.collision_weight = collision_weight
        self.time_weight = time_weight
        self.dist_weight = dist_weight
        self.exploration_weight = exploration_weight
        self.targets_collected_weight = targets_collected_weight
        self.speed_weight = speed_weight

    def linar_fitness(self, num_collisions: int,
            num_time_steps: int,
            dist_to_target: float,
            map_unexplored: float,
            speed: float,
            targets_collected: float) -> float:
    
        return (self.collision_weight * num_collisions
                + self.time_weight * num_time_steps
                + self.dist_weight * dist_to_target
                + self.exploration_weight * map_unexplored
                + self.targets_collected_weight * targets_collected
                + self.speed_weight * speed
                )

def distance_to_target(target, position):
    return math.sqrt((target[0] - position[0]) ** 2 + (target[1] - position[1]) ** 2)

# discretize sensor value into a given number of discrete values within a range
def real_to_binary(value, number_of_discrete_values, min_value, max_value):
    index = 0
    step_size = (abs(max_value-min_value)) / number_of_discrete_values
    for i in range(number_of_discrete_values):
        if value < min_value + step_size * (i+1): 
            index = i 
            break
        
    def int_to_binary(n, number_of_discrete_values):
        binary = ""
        while n > 0:
            binary = str(n % 2) + binary
            n //= 2
        difference = math.log2(number_of_discrete_values) - len(binary)
        if difference != 0: 
            for i in range(int(difference)):
                binary = "0" + binary
        return binary
    return int_to_binary(index, number_of_discrete_values)

def compute_map_exploration(grid_prob: np.ndarray, threshold: float = 1e-7):
    deviation = np.abs(grid_prob - 0.5)
    explored = deviation > threshold
    return (1 - np.sum(explored) / grid_prob.size)
