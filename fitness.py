import math
from numpy import log2, number

# fitness function to evaluate phenome and control evolutionary algorithm
def fitness (number_collision, number_time_steps, distance_to_target, collision_weight, time_weight, distance_weight):
    
    # Calculate the fitness score
    fitness_score = (
        collision_weight * number_collision +
        time_weight * number_time_steps +
        distance_weight * distance_to_target
    )

    return fitness_score

# calculate euclidean distance from robot's real postion to target position
def distance_to_target (target, position):
    """
    Calculate the euclidean distance to the target.
    """
    distance = ((target[0] - position[0]) ** 2 + (target[1] - position[1]) ** 2) ** 0.5
    return distance

# discretize sensor value into a given number of discrete values within a range
def real_to_binary(value, number_of_discrete_values, min_value, max_value):
   
    index = 0

    step_size = (abs(max_value-min_value))/number_of_discrete_values
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

