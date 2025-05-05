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
from robot import Robot
import random

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

# map resolution for occupancy grid
PAD = 20
BLOCK_W, BLOCK_H = 16, 9
BLOCK_SIZE = 100
SCREEN_W, SCREEN_H = BLOCK_W * BLOCK_SIZE, BLOCK_H * BLOCK_SIZE
WALL_THICKNESS = 4
GRID_SIZE = WALL_THICKNESS

# the drawable map area (inside the padding)
MAP_W = SCREEN_W - 2 * PAD
MAP_H = SCREEN_H - 2 * PAD

# occupancy grid: cols = MAP_W/GRID_SIZE, rows = MAP_H/GRID_SIZE
grid = np.zeros((
    int(SCREEN_W  / GRID_SIZE),
    int(SCREEN_H  / GRID_SIZE)
))

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

# near the top of your file, pick a wall probability
WALL_H_PROB = 0.2
WALL_V_PROB = 0.2

# then when you initialize:
horiz = [
    [1 if random.random() < WALL_H_PROB else 0
     for _ in range(BLOCK_W)]
    for _ in range(BLOCK_H+1)
]
vert = [
    [1 if random.random() < WALL_V_PROB else 0
     for _ in range(BLOCK_W+1)]
    for _ in range(BLOCK_H)
]

# force all outer borders ON
# top and bottom horizontal walls:
for c in range(BLOCK_W):
    horiz[0][c] = 1             # top edge of map
    horiz[BLOCK_H][c] = 1        # bottom edge of map

# left and right vertical walls:
for r in range(BLOCK_H):
    vert[r][0] = 1              # left edge of map
    vert[r][BLOCK_W] = 1         # right edge of map

# Game Loop 
running = True
while running:

    #check for close game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # reset screen
    screen.fill((255,255,255))

    walls = draw_map(
        screen,
        horiz, vert,
        pad=PAD,
        grid_w=BLOCK_W, grid_h=BLOCK_H,
        wall_color=(0,0,0),
        wall_thickness=4
    )
    landmarks = compute_landmarks(
        horiz, vert,
        screen,
        pad=PAD,
        grid_w=BLOCK_W, grid_h=BLOCK_H
    )
    for (i, m_x, m_y) in landmarks:
         pygame.draw.circle(screen, "blue", (m_x,m_y), 5)

    robot.draw_Robot(screen)
    robot.sense(walls, screen, draw_sensors)

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

    # Draw the robot's trajectory
    if draw_estimated_path:
        robot.drawTrajectories(screen)

    # Add uncertainty ellipse to the robot in intervals of SAMPLE_INTERVAL
    now = pygame.time.get_ticks()
    if now - last_sample_time >= SAMPLE_INTERVAL:
        # grab current estimate
        mean = kf.mu.copy()          # [x, y, θ]
        cov  = kf.sigma[:2, :2].copy()  # only the x–y block
        robot.uncertainty_regions.append((mean, cov))
        last_sample_time = now
    if draw_sigma:
        robot.draw_uncertainty_ellipse(screen)
    
    free_cells, occipied_cells = get_observed_cells(robot, GRID_SIZE, grid.shape[0], grid.shape[1])
    
    if draw_observed_cells:
        draw_cells(free_cells, screen, GRID_SIZE)
        draw_cells(occipied_cells, screen, GRID_SIZE, "green")

    for free in free_cells:
        grid[free[0]][free[1]] += -0.85

    for occ in occipied_cells:
        grid[occ[0]][occ[1]] += 2.2

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] > 0:
                draw_cells_filled([(i, j)], screen, GRID_SIZE, "purple")


    # Shows on display
    pygame.display.flip()

    clock.tick(60)

pygame.quit()