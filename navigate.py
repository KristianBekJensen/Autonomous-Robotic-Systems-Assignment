import numpy as np
import pygame

from fitness import real_to_binary
from leap_ec.binary_rep.decoders import BinaryToIntDecoder

def navigate(keys, v_left, v_right):
    if keys[pygame.K_w]:
        if v_left != v_right:
            v_right = max(v_right, v_left)
            v_left = max(v_right, v_left)
        else:
            v_right += 0.2
            v_left += 0.2
    elif keys[pygame.K_s]:
        v_left = max(0, v_left - 0.2)
        v_right = max(0, v_right - 0.2)
    elif keys[pygame.K_d]:
        v_right *= 0.98
    elif keys[pygame.K_a]:
        v_left *= 0.98
    elif keys[pygame.K_e]:
        if v_left == 0 and v_right == 0:
            v_left = 0.3
            v_right = -v_left
    elif keys[pygame.K_q]:
        if v_left == 0 and v_right == 0:
            v_right = 0.3
            v_left = -v_right
    else:
        if v_right < 0 or v_left < 0:
            v_right = v_left = 0
        elif v_right >= v_left:
            v_left = v_right
        else:
            v_right = v_left
    return v_left, v_right

def phenome_navigate(phenome, sensor_vals, v_left, v_right, min_speed, max_speed, angle_to_path, wheel_inputs, angle_inputs):
    sensor_threshold = 50
    
    strBits = ""
    for dis in sensor_vals:
        if dis < sensor_threshold:
            strBits += "1"
        else:
            strBits += "0"
    single_wheel_inputs = int(2**(wheel_inputs/2))
    strBits += real_to_binary(v_left, single_wheel_inputs, min_speed, max_speed)
    strBits += real_to_binary(v_right, single_wheel_inputs, min_speed, max_speed)
    strBits += real_to_binary(angle_to_path, 2**angle_inputs, 0 , np.pi * 2)
    
    #Convert from bits to Interger
    bits = []
    for bit in strBits:
        bits.append(int(bit))
    d = BinaryToIntDecoder(len(bits))
    index_of_action = d.decode(np.array(bits))[0]
    
    #Find location of genome instructions. * 2 because each intstruction is 2 bits
    index_of_action = (phenome[index_of_action * 2], phenome[(index_of_action * 2) + 1])
    # Excute instruction either accelrate, decellarate, turn left or turn right
    if index_of_action == (False, True):
        if v_left != v_right:
            v_right = max(v_right, v_left)
            v_left = max(v_right, v_left)
        else:
            v_right += 0.2
            v_left += 0.2
    elif index_of_action == (False, False):
            v_left = max(1, v_left - 0.2)
            v_right = max(1, v_right - 0.2)
    elif index_of_action == (True, False):
            v_right *= 0.98
    elif index_of_action == (True, True):
            v_left *= 0.98
    else:
        print("Something went terrebly wrong")

    return np.median([min_speed, v_left, max_speed]), np.median([min_speed, v_right, max_speed])