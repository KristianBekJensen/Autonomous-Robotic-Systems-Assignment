import math
from numpy import log2, number

def fitness(num_collision, num_time_steps, dist_to_target):
    coll_w, time_w, dist_w = 100, 1, 1
    fitness_score = ((coll_w * num_collision) + (time_w * num_time_steps) + (dist_w * dist_to_target))
    return fitness_score

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

