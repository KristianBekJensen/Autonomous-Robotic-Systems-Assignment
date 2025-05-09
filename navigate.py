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

def phenome_navigate(phenome, sensor_vals, v_left, v_right):
    sensor_threshold = 50
    k = ""
    for dis in sensor_vals:
        if dis < sensor_threshold:
            k += "1"
        else:
            k += "0"
    k += real_to_binary(v_left, 8, 1, 3)
    k += real_to_binary(v_right, 8, 1, 3)

    bits = []
    for k in k:
        bits.append(int(k))
    bits = numpy.array(bits)
    d = BinaryToIntDecoder(len(bits))
    key = d.decode(bits)[0]
    key = key * 2
    # print("keys", phenome[key],  phenome[key+1])
    if phenome[key] == 0:
        if phenome[key+1] == 1:
            if v_left != v_right:
                v_right = max(v_right, v_left)
                v_left = max(v_right, v_left)
            else:
                v_right += 0.2
                v_left += 0.2
        elif phenome[key+1] == 0:
            v_left = max(1, v_left - 0.2)
            v_right = max(1, v_right - 0.2)

    elif phenome[key] == 1:
        if phenome[key+1] == 0:
            v_right *= 0.98
        elif phenome[key+1] == 1:
            v_left *= 0.98

    return min(3, v_left), min(v_right, 3)