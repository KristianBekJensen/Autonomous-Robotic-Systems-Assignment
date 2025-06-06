import numpy as np
import pygame

def navigate(keys, v_left, v_right):
    """ taking in keyboard commands and changing left/right wheel speeds
    based on key for manual control of the robot in the map """

    # w -> accelerate forward
    if keys[pygame.K_w]:
        if v_left != v_right:
            v_right = max(v_right, v_left)
            v_left = max(v_right, v_left)
        else:
            v_right += 0.2
            v_left += 0.2
    # s -> decelerate
    elif keys[pygame.K_s]:
        v_left = max(0, v_left - 0.2)
        v_right = max(0, v_right - 0.2)
    # d -> steer right
    elif keys[pygame.K_d]:
        v_right *= 0.98
    # a -> steer left
    elif keys[pygame.K_a]:
        v_left *= 0.98
    # e -> rotate clockwise (only when speed = 0)
    elif keys[pygame.K_e]:
        if v_left == 0 and v_right == 0:
            v_left = 0.3
            v_right = -v_left
    # q -> rotate counter clockwise (only when speed = 0)
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