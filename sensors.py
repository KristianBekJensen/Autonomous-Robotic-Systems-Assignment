import pygame
import math
import sys
import numpy as np

def draw_sensors(screen, robot_state, sensor_length, draw=False):
    x, y, theta = robot_state
    sensor_lines = []
    for i in range(12):
        angle = theta + (i * (math.pi / 6))
        dx = sensor_length * math.cos(angle)
        dy = sensor_length * math.sin(angle)
        start = (x, y)
        end = (x + dx, y + dy)
        sensor_lines.append([i, (start, end)])
        if draw:
            pygame.draw.line(screen, "black", start, end)
    return sensor_lines

def detect_walls(sensor_lines, wall, robot_state, robot_radius):
    x_robot, y_robot, theta_robot = robot_state
    collided_lines = np.full(len(sensor_lines), 100-robot_radius)
    for i, line in sensor_lines:
        start, end = line
        clipped_line = wall.clipline(start, end)
        if clipped_line:
            start, end = clipped_line
            x, y = start
            dist = math.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2) - robot_radius
            collided_lines[i] = dist
    return collided_lines

def closest_wall(sensor_line, walls, robot_state):
    x_robot, y_robot, theta_robot = robot_state
    start, end = sensor_line
    closest_wall = 0
    distance_to_wall = sys.float_info.max
    for wall in walls:
        clipped_line = wall.clipline(start, end)
        if clipped_line:
            start, end = clipped_line
            x, y = start
            dist = math.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
            if dist < distance_to_wall:
                closest_wall = wall
    
    return closest_wall