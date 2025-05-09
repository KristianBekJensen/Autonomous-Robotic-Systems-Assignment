import numpy
import pygame

from fitness import fitness, real_to_binary
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

def phenome_navigate(phenome, sensor_vals, v_left, v_right, min_speed, max_speed):
    sensor_threshold = 50
    
    strBits = ""
    for dis in sensor_vals:
        if dis < sensor_threshold:
            strBits += "1"
        else:
            strBits += "0"
    strBits += real_to_binary(v_left, 8, min_speed, max_speed)
    strBits += real_to_binary(v_right, 8, min_speed, max_speed)
    
    #Convert from bits to Interger
    bits = []
    for bit in strBits:
        bits.append(int(bit))
    d = BinaryToIntDecoder(len(bits))
    key = d.decode(numpy.array(bits))[0]
    
    #Find location of genome instructions. * 2 because each intstruction is 2 bits
    key = (phenome[key * 2], phenome[(key * 2) + 1])
    # Excute instruction either accelrate, decellarate, turn left or turn right
    if key == (False, True):
        if v_left != v_right:
            v_right = max(v_right, v_left)
            v_left = max(v_right, v_left)
        else:
            v_right += 0.2
            v_left += 0.2
    elif key == (False, False):
            v_left = max(1, v_left - 0.2)
            v_right = max(1, v_right - 0.2)
    elif key == (True, False):
            v_right *= 0.98
    elif key == (True, True):
            v_left *= 0.98
    else:
        print("Something went terrebly wrong")

    return numpy.median([min_speed, v_left, max_speed]), numpy.median([min_speed, v_right, max_speed])