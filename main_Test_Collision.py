import pygame
import math
import numpy as np
import itertools
from collections import deque
from kinematics import differential_drive_kinematics
from mapping import get_observed_cells, line_through_grid
from navigate import navigate
import sensors as sn
import maps
from kalman_filter import KalmanFilter
from landmark import *
from utils import draw_covariance_ellipse, draw_dashed_lines
from map_Refactor import *
from robotV2 import Robot

# pygame setup
pygame.init()

default_font = pygame.font.SysFont('Arial', 10)

draw_observed_cells = False
draw_landmark_line = False
draw_sigma = False
draw_estimated_path = False
draw_sensors = False

#set up display 
clock = pygame.time.Clock()

# Setup Robot Parameter
theta = 0
x = 100
y = 100
v_left = 0
v_right = 0
r = 20 # robot radius
initial_pose = np.array([x, y, theta])
axel_length = 15 # distance between wheels
max_sensor_range = 500 # max sensor range
num_sensors = 96

robot = Robot(x,y,theta,r, axel_length, max_sensor_range, num_sensors) # rn generic parameters match above, apply changes if needed 

# Setup Kalman Filter
process_noise = 0.1
position_measurement_noise = 1
theta_mesurement_noise = 0.5
R = np.diag([2, 2, 0.2**2])  # Process noise
Q = np.diag([2, 2, 0.2**2])  # Measurement noise
initial_covariance = np.diag([0.1, 0.1, 0.1])  # For local localization
SAMPLE_INTERVAL = 2000
last_sample_time = pygame.time.get_ticks()


kf = KalmanFilter(initial_pose, initial_covariance, R, Q)


# Map and Screen Setup
N_X = 20
N_Y = 10
BLOCK_WIDTH = 80
BLOCK_HEIGHT = BLOCK_WIDTH
WALL_THICKNESS = 4
MAP_WIDTH = N_X * BLOCK_WIDTH
MAP_HEIGHT = N_Y * BLOCK_HEIGHT

# Mapping Setup
GRID_SIZE = WALL_THICKNESS
grid = np.zeros((int(MAP_WIDTH/GRID_SIZE), int(MAP_HEIGHT/GRID_SIZE)))     

screen = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
blocks = generate_sample_map(N_X, N_Y)
walls, landmarks = draw_map(screen, blocks, wall_thickness=WALL_THICKNESS, block_width=BLOCK_WIDTH, block_height=BLOCK_HEIGHT)

# Game Loop 
running = True
while running:

    #check for close game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # reset screen
    screen.fill((255,255,255))

    draw_map(screen, blocks, wall_thickness=WALL_THICKNESS, block_width=BLOCK_WIDTH, block_height=BLOCK_HEIGHT)
    for (i,m_x,m_y) in landmarks:
        pygame.draw.circle(screen, "blue", (m_x,m_y), 5)

    robot.draw_Robot(screen)
    #robot.sense(walls, screen, draw_sensors)

    # detect landmarks
    detected_landmarks = robot.detect_landmarks(landmarks, screen, draw_landmark_line)
    robot.estimate_pose(kf, landmarks, detected_landmarks, screen, position_measurement_noise, theta_mesurement_noise, process_noise)

    # React on key inputs from the user and adjust wheel speeds
    keys = pygame.key.get_pressed()
    v_left, v_right = navigate(keys, robot.v_left, robot.v_right)
    robot.v_left = v_left
    robot.v_right = v_right

    # Move the robot and execute collision handling
    robot.move(walls)


    # Shows on display
    pygame.display.flip()

    clock.tick(60)

pygame.quit()