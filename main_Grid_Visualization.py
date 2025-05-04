from os import truncate
import pygame
import math
import numpy as np
import itertools
from collections import deque
from kinematics import differential_drive_kinematics
from mapping import get_observed_cells, line_through_grid, log_odds_to_prob, probs_to_grey_scale
from navigate import navigate
import sensors as sn
import maps
from kalman_filter import KalmanFilter
from landmark import *
from utils import draw_covariance_ellipse, draw_dashed_lines
from map_Refactor import *
from robot import Robot

# pygame setup
pygame.init()

default_font = pygame.font.SysFont('Arial', 10)

draw_observed_cells = True
draw_landmark_line = False
draw_sigma = False
draw_estimated_path = False
draw_sensors = False

# Setup Robot Parameter
theta = 0
x = 100
y = 100
v_left = 0
v_right = 0
r = 20  # robot radius
initial_pose = np.array([x, y, theta])
axel_length = 15
max_sensor_range = 200
num_sensors = 96

robot = Robot(x, y, theta, r, axel_length, max_sensor_range, num_sensors)

# Setup Kalman Filter
process_noise = 0.1
position_measurement_noise = 1
theta_mesurement_noise = 0.5
R = np.diag([2, 2, 0.2**2])
Q = np.diag([2, 2, 0.2**2])
initial_covariance = np.diag([0.1, 0.1, 0.1])
SAMPLE_INTERVAL = 2000
last_sample_time = pygame.time.get_ticks()

kf = KalmanFilter(initial_pose, initial_covariance, R, Q)

# Map and Screen Setup
N_X = 10
N_Y = 10
BLOCK_WIDTH = 80
BLOCK_HEIGHT = BLOCK_WIDTH
WALL_THICKNESS = 4
MAP_WIDTH = N_X * BLOCK_WIDTH
MAP_HEIGHT = N_Y * BLOCK_HEIGHT
GRID_SIZE = WALL_THICKNESS

# Allocate grid
grid = np.zeros((int(MAP_WIDTH / GRID_SIZE), int(MAP_HEIGHT / GRID_SIZE)))
grid_probability = np.full_like(grid, 0.5)

# Set up combined screen (left: simulation, right: visualization)
TOTAL_WIDTH = MAP_WIDTH * 2
TOTAL_HEIGHT = MAP_HEIGHT
screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
main_surface = screen.subsurface((0, 0, MAP_WIDTH, MAP_HEIGHT))
second_surface = screen.subsurface((MAP_WIDTH, 0, MAP_WIDTH, MAP_HEIGHT))

blocks = generate_sample_map(N_X, N_Y)
walls, landmarks = draw_map(main_surface, blocks, wall_thickness=WALL_THICKNESS,
                            block_width=BLOCK_WIDTH, block_height=BLOCK_HEIGHT)

# Game Loop
clock = pygame.time.Clock()
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Reset screens
    main_surface.fill((255, 255, 255))
    second_surface.fill((255, 255, 255))

    # Draw main map and landmarks
    draw_map(main_surface, blocks, wall_thickness=WALL_THICKNESS,
             block_width=BLOCK_WIDTH, block_height=BLOCK_HEIGHT)

    for (i, m_x, m_y) in landmarks:
        pygame.draw.circle(main_surface, "blue", (m_x, m_y), 5)

    robot.draw_Robot(main_surface)
    robot.sense(walls, main_surface, draw_sensors)

    # Detect landmarks
    detected_landmarks = robot.detect_landmarks(landmarks, main_surface, draw_landmark_line)
    robot.estimate_pose(kf, landmarks, detected_landmarks, main_surface,
                        position_measurement_noise, theta_mesurement_noise, process_noise)

    # Handle keypress navigation
    keys = pygame.key.get_pressed()
    v_left, v_right = navigate(keys, robot.v_left, robot.v_right)
    robot.v_left = v_left
    robot.v_right = v_right

    robot.move(walls)

    if draw_estimated_path:
        robot.drawTrajectories(main_surface)

    now = pygame.time.get_ticks()
    if now - last_sample_time >= SAMPLE_INTERVAL:
        mean = kf.mu.copy()
        cov = kf.sigma[:2, :2].copy()
        robot.uncertainty_regions.append((mean, cov))
        last_sample_time = now

    if draw_sigma:
        robot.draw_uncertainty_ellipse(main_surface)

    # Mapping
    free_cells, occipied_cells = get_observed_cells(robot, GRID_SIZE,
                                                    int(MAP_WIDTH / GRID_SIZE), int(MAP_HEIGHT / GRID_SIZE))

    if draw_observed_cells:
        draw_cells(free_cells, main_surface, GRID_SIZE, "red")
        draw_cells(occipied_cells, main_surface, GRID_SIZE, "green")

    # Update the log odds depending on occupied and free cells 
    for free in free_cells:
        grid[free[0]][free[1]] += -0.1

    for occ in occipied_cells:
        grid[occ[0]][occ[1]] += 0.2

    grid_probability = log_odds_to_prob(grid)
    grid_probability = probs_to_grey_scale(grid_probability)

    # TODO: fix the grids outside of the sensor range 
    


    # Draw grid probabilities to second screen
    for i in range(grid_probability.shape[0]):
        for j in range(grid_probability.shape[1]):
            grey = grid_probability[i][j]
            color = (grey, grey, grey)
            rect = pygame.Rect(i * GRID_SIZE, j * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(second_surface, color, rect)

    # Refresh screen
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
