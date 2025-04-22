import pygame
import math
import numpy as np
import itertools
from collections import deque
from kinematics import differential_drive_kinematics
from navigate import navigate
import sensors as sn
import maps
from kalman_filter import KalmanFilter
from landmark import *
from utils import draw_covariance_ellipse, draw_dashed_lines

# pygame setup
pygame.init()

default_font = pygame.font.SysFont('Arial', 14)

debug = True

#set up display 
display_width = 720
display_height = 1000
screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
dt = 0
r = 30 # robot radius

# Setup Robot intial Pos
theta = 0
x = 100
y = 100
v_left = 0
v_right = 0

# Define landmarks as (i, x, y) with i as signature
landmarks = [
    (0, 0, 0), 
    (1, 720, 0), 
    (2, 720, 1000), 
    (3, 0, 1000),
    (4, 0, 200),
    (5, 480, 200),
    (6, 240, 400), 
    (7, 720, 400),
    (8, 240, 800),
    (9, 480, 600),
    (10, 480, 1000),
]

# how often to draw an uncertainty region (ms)
SAMPLE_INTERVAL = 2000  
last_sample_time = pygame.time.get_ticks()

# store (mean, cov) tuples
uncertainty_regions = []

def draw_landmarks(screen, landmarks):
    for (i, m_x, m_y) in landmarks:
        pygame.draw.circle(screen, (0, 100, 255), (m_x, m_y), 5)  # Blue landmark dot
        # text_surface = default_font.render(f"LM{i}", True, (0, 0, 0))
        # screen.blit(text_surface, (m_x + 8, m_y - 8))

max_sensor_range = 250 - r
senors_values = np.full(12, max_sensor_range)

def detect_landmarks(theta):
    detected_landmarks = []
    for landmark in landmarks: 
        i, m_x, m_y = landmark
        distance = math.sqrt((x - m_x) ** 2 + (y - m_y) ** 2)

        if distance < max_sensor_range:
            pygame.draw.line(screen, (0, 255, 0), (x, y), (m_x, m_y), 2)
            detected_landmarks.append(landmark)

    return detected_landmarks

initial_pose = np.array([x, y, theta])
true_pose = initial_pose.copy()

# Define Real noise
process_noise = 0.1
position_measurement_noise = 1
theta_mesurement_noise = 0.5
# Initialize Kalman
R = np.diag([2, 2, 0.2**2])  # Process noise
Q = np.diag([2, 2, 0.2**2])  # Measurement noise

# For local localization, start with high confidence (small covariance)
# For global localization, start with low confidence (large covariance)
initial_covariance = np.diag([0.1, 0.1, 0.1])  # For local localization
# initial_covariance = np.diag([10.0, 10.0, 10.0])  # For global localization

kf = KalmanFilter(initial_pose, initial_covariance, R, Q)

true_poses = []
estimated_poses = []
uncertainty_regions = []

def point_on_circles_circumference(x, y, r, theta, angle):
    angle = (math.degrees(theta) + angle) % 360
    angle = math.radians(angle)
    
    x2 = x + np.cos(angle) * r
    y2 = y + np.sin(angle) * r

    return x2, y2

def draw_robot_text(x, y, r, theta, angle, text):
    x2, y2 = point_on_circles_circumference(x, y, r, theta, angle)
    
    # Adjust coords from center of textbox to coords topleft of textbox
    x2-=7
    y2-=7

    text_surface = default_font.render(text, False, (0, 0, 0))
    screen.blit(text_surface, (x2,y2))

# Checks for what side robot was previously on and and puts the new coords on the correct side
def check_x_wall(wall, new_x, x):
    if x <= wall.x: # Robot left
        new_x = wall.x - r
    elif x >= wall.x + wall.width: # Robot right
        new_x = wall.x + wall.width + r
    return new_x

def check_y_wall(wall, new_y, y):
    if y <= wall.y: # Robot above
        new_y = wall.y - r
    elif y >= wall.y + wall.height: # Robot below
        new_y = wall.y + wall.height + r
    return new_y



running = True
while running:

    #check for close game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # fill the screen with a color to wipe away anything from last frame
    screen.fill((255,255,255))

    # Define the edges for the map (creates a 980x980 area with a 10 pixel margin).
    edges = [(0, 0), (720, 0), (720, 1000), (0, 1000)]

    # Define some walls as an example.
    wall_coords = {
        "1": ((0, 200), (480, 200)),
        "2": ((240, 400), (720, 400)),
        "3": ((240, 400), (240, 800)),
        "4": ((480, 600), (480, 1000)),
    }
    wall_thickness = 3
    walls = maps.draw_map(screen, edges, wall_coords, wall_thickness)
    
    # Draw landmarks on the map
    draw_landmarks(screen, landmarks)
    # Detect landmarks 
    detected_landmarks = detect_landmarks(theta)

    # calculate new (potential) state
    state = [x, y, theta]
    state_change = differential_drive_kinematics(state, v_left, v_right)
    new_x = x + state_change[0]
    new_y = y + state_change[1]
    new_theta = (theta + state_change[2]) % (2 * math.pi)


    # rectangle representation of the robot
    new_robot_rect = pygame.Rect(new_x - r, new_y - r, 2 * r, 2 * r) 
    if debug:
        pygame.draw.circle(screen, "green", (new_x, new_y), r) # Robot if theres no collosions

    # Checks for collosions between new postion and old postion
    colliding_walls = []
    start_x, start_y = point_on_circles_circumference(x, y, r, theta, 90)
    end_x, end_y = point_on_circles_circumference(new_x, new_y, r, new_theta, 90)

    right_clossion = sn.closest_wall(((start_x, start_y),(end_x, end_y)), walls, state)
    if right_clossion != 0:
        if debug:
            pygame.draw.rect(screen, "yellow", right_clossion)
        colliding_walls.append(right_clossion)
    if debug:
        pygame.draw.line(screen, "purple", (start_x, start_y), (end_x, end_y))
    
    start_x, start_y = point_on_circles_circumference(x, y, r, theta, 270)
    end_x, end_y = point_on_circles_circumference(new_x, new_y, r, new_theta, 270)

    left_clossion = sn.closest_wall(((start_x, start_y),(end_x, end_y)), walls, state)
    if left_clossion != 0:
        if debug:
            pygame.draw.rect(screen, "blue", left_clossion)
        colliding_walls.append(left_clossion)
    if debug:
        pygame.draw.line(screen, "purple", (start_x, start_y), (end_x, end_y))    
    
    # Check for collisions with new postion
    for wall in walls:
        if wall.colliderect(new_robot_rect):
            colliding_walls.append(wall)
    
    # Check all colliding walls to find new coords
    if len(colliding_walls) != 0:
        if len(colliding_walls) == 1:
            new_x = check_x_wall(colliding_walls[0], new_x, x)
            new_y = check_y_wall(colliding_walls[0], new_y, y)
        else:
            for wall in colliding_walls:
                if wall.width > wall_thickness:
                        new_y = check_y_wall(wall, new_y, y)
                else:
                        new_x = check_x_wall(wall, new_x, x)

    # Update state variables
    x, y, theta = new_x, new_y, new_theta
    robot_pose = np.array([x, y, theta])
    true_poses.append(robot_pose)

    # # Draw the true pose trajectory as a solid line
    if len(true_poses) > 1:
        pts = [(px, py) for px, py, _ in true_poses]
        pygame.draw.lines(screen, (0, 0, 0), False, pts, 2)

    # Kalman Filter part
    v = (v_left + v_right) / 2
    omega = state_change[2]
    u = np.array([v, omega])

    # Draw the robot and it's direction line
    pygame.draw.circle(screen, "black", (x, y), r) # Draw robot
    pygame.draw.circle(screen, "white", (x, y), r - 2) # Draw robot
    pygame.draw.line(screen, (140, 0, 140), (x, y), (np.cos(theta) * r + x, np.sin(theta) * r + y), 3)
    
    estimated_pose, sigma = kf.predict(u, 1, process_noise)
    
    # Update estimation based on landmarks
    measurements = get_landmark_measurements(detected_landmarks, robot_pose)
    num_detected_landmarks = len(measurements)
    if num_detected_landmarks == 2:
        p1, p2, estimate_theta = two_point_triangulate(measurements, landmarks, robot_pose, position_measurement_noise, theta_mesurement_noise)
        if p1 != (0,0):
            if debug:
                pygame.draw.circle(screen, "purple", p1, 10)
                pygame.draw.circle(screen, "green", p2, 10)
            z = [p1[0], p1[1], estimate_theta]
            estimated_pose, sigma = kf.update(z)
    # more than 2 detected landmarks
    elif num_detected_landmarks > 2:
        zs = []
        avg_theta = 0
        # for every unique pair of landmark measurements
        for i, j in itertools.combinations(range(num_detected_landmarks), 2):
            meas_pair = [measurements[i], measurements[j]]
            p1, p2, estimate_theta = two_point_triangulate(meas_pair, landmarks, robot_pose, position_measurement_noise, theta_mesurement_noise)
            if debug:
                pygame.draw.circle(screen, "purple", p1, 10)
                pygame.draw.circle(screen, "green", p2, 10)
            # keep only the true intersection
            if p1 != (0, 0):
                avg_theta += estimate_theta
                zs.append(p1)

        if zs:
            # average all the valid p1’s
            avg_x = sum(pt[0] for pt in zs) / len(zs)
            avg_y = sum(pt[1] for pt in zs) / len(zs)
            z = [avg_x, avg_y, avg_theta/len(zs)]
            estimated_pose, sigma = kf.update(z)
        
    estimated_poses.append(estimated_pose)
    
    now = pygame.time.get_ticks()
    if now - last_sample_time >= SAMPLE_INTERVAL:
        # grab current estimate
        mean = kf.mu.copy()          # [x, y, θ]
        cov  = kf.sigma[:2, :2].copy()  # only the x–y block
        uncertainty_regions.append((mean, cov))
        last_sample_time = now
    
    for mean, cov in uncertainty_regions:
        draw_covariance_ellipse(
            screen,
            mean=(mean[0], mean[1]),
            cov=cov,
            n_std=4.0,           
            num_points=64,
            color="orange",
            width=2
        )
    
    # Draw the estimated pose trajectory as a dotted line
    if len(estimated_poses) > 1:
        pts = [(px, py) for px, py, _ in estimated_poses]
        pygame.draw.lines(screen, "red", False, pts, 2)

    # Draw sensors
    state = (x, y, theta)
    sensor_lines = sn.draw_sensors(screen, state, 100, False)

    # Detect walls with sensors
    activatedSensors = []
    for wall in walls:
        new_senors_values =  sn.detect_walls(sensor_lines, wall, state, r)
        for i in range(len(senors_values)):
            if new_senors_values[i] < max_sensor_range:
                senors_values[i] = new_senors_values[i]
                activatedSensors.append(i)
    for i in range(12):
        if i not in activatedSensors:
            senors_values[i] = max_sensor_range
    
                
    # React on key inputs from the user and adjust wheel speeds
    keys = pygame.key.get_pressed()
    v_left, v_right = navigate(keys, v_left, v_right)
    
    draw_robot_text(x, y, r/3, theta, 270, str(v_left.__round__(1))) # draw left wheel speed
    draw_robot_text(x, y, r/3, theta, 90, str(v_right.__round__(1))) # draw right wheel speed
    
    # for i in range(12): # draw sensor values
    #     draw_robot_text(x, y, r+10, theta, 30*i, str(round(senors_values[i], 1)))

    # Shows on display
    pygame.display.flip()
    dt = clock.tick(60)

pygame.quit()