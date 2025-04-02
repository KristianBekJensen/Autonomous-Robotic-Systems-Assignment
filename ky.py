import pygame
import math
import numpy as np
from scipy.integrate import odeint

""" robot_image = pygame.image.load("assets/robot_model.png")

original_width, original_height = robot_image.get_size()
desired_width = 200  
desired_height = int(desired_width * original_height / original_width)  # Maintain aspect ratio
robot_image = pygame.transform.smoothscale(robot_image, (desired_width, desired_height))
 """
# Robot parameters
wheel_radius = 0.05
axel_lenght = 3

def differential_drive_kinematics(state, t, v_left, v_right):
    x, y, direction = state
    v = (v_left + v_right)/2 
    rotation = (v_right - v_left)/axel_lenght 
    print(direction)
    dxdt = v*np.cos(direction)
    dydt = v*np.sin(direction)

    return [dxdt, dydt, rotation]


def update_robot_state(initial_state, t, v_left, v_right):
    return odeint(differential_drive_kinematics, initial_state, t, args=(v_left, v_right))


""" def draw_robot(screen, x, y, theta):
    rotated_image = pygame.transform.rotate(robot_image, -math.degrees(theta))
    new_rect = rotated_image.get_rect(center=(x, y))
    screen.blit(rotated_image, new_rect.topleft)
 """