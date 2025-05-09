import pygame
import numpy as np
import matplotlib.pyplot as plt
from mapping import calculate_mapping_accuracy, get_observed_cells, log_odds_to_prob, probs_to_grey_scale
from navigate import navigate
from kalman_filter import KalmanFilter
from maps import *
from robot import Robot
# Import the trajectory recorder
from trajectory_recorder import TrajectoryRecorder
from path_finder import *

# pygame setup
pygame.init()

default_font = pygame.font.SysFont('Arial', 10)
info_font = pygame.font.SysFont('Arial', 18)

# set debugging booleans
(draw_observed_cells, draw_landmark_line, draw_sigma, draw_sensors) = (False,) * 4
draw_estimated_path = True
visualize_mapping = True

# Create trajectory recorder
trajectory_recorder = TrajectoryRecorder()

#set up display 
clock = pygame.time.Clock()

# robot params
x, y, theta = 100, 100, 0 
initial_pose = np.array([x, y, theta])
v_left, v_right = 0, 0
r = 20 # robot radius
axel_length = 15

# sensor params
max_sensor_range, num_sensors = 100, 48
sensor_noise = 0

robot = Robot(x, y, theta, r, axel_length, max_sensor_range, num_sensors) # rn generic parameters match above, apply changes if needed 

# Setup Kalman Filter
process_noise = 0.01
position_measurement_noise = 0.01
theta_mesurement_noise = 0.005
R = np.diag([process_noise, process_noise, theta_mesurement_noise])  # Process noise
Q = np.diag([position_measurement_noise, position_measurement_noise, theta_mesurement_noise])  # Measurement noise
initial_covariance = np.diag([0.1, 0.1, 0.1])  # For local localization
SAMPLE_INTERVAL = 2000
last_sample_time = pygame.time.get_ticks()

kf = KalmanFilter(initial_pose, initial_covariance, R, Q)  

# map resolution for occupancy grid
PAD = 20
NUM_BLOCKS_W, NUM_BLOCKS_H = 8, 8
BLOCK_SIZE = 100
SCREEN_W, SCREEN_H = NUM_BLOCKS_W * BLOCK_SIZE, NUM_BLOCKS_H * BLOCK_SIZE
WALL_THICKNESS = 4
GRID_SIZE = WALL_THICKNESS

# occupancy grid: cols = MAP_W/GRID_SIZE, rows = MAP_H/GRID_SIZE
grid = np.zeros((
    int(SCREEN_W  / GRID_SIZE),
    int(SCREEN_H  / GRID_SIZE)
))
# occupancy grid as probabilites [0;1]
grid_probability = np.full((
    int(SCREEN_W / GRID_SIZE),
    int(SCREEN_H / GRID_SIZE)
), 0.5)

sensor_noise = 0

screen = pygame.display.set_mode((2*SCREEN_W, SCREEN_H))
main_surface = screen.subsurface((0,0,SCREEN_W, SCREEN_H))
if visualize_mapping:
    second_surface = screen.subsurface((SCREEN_W, 0, SCREEN_W, SCREEN_H))

# All Map Elements
walls, landmarks, obstacles = draw_map(
    main_surface, num_blocks_w=NUM_BLOCKS_W, num_blocks_h=NUM_BLOCKS_H, 
    pad=PAD, 
    wall_color=(0, 0, 0), 
    wall_h_prob=0.2,
    wall_v_prob=0.2,
    wall_thickness=WALL_THICKNESS,
    p_landmark=0.25,
    n_obstacles=50,
    obstacle_mu=7.5,
    obstacle_sigma=1.5,
    obstacle_color=(0,0,0)
)

# Trajectory mode info
mode_info = "Manual Control"
mode_color = (0, 0, 255)  # Blue for manual control

# Variable for trajectory filename
trajectory_filename = "trajectory.pkl"

pygame.display.flip()

values = []
# Game Loop 
running = True
while running:
    current_time = pygame.time.get_ticks()
    
    #check for close game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Trajectory recording/playback controls
            if event.key == pygame.K_r:  # R key - Start/Stop Recording
                if trajectory_recorder.is_recording():
                    trajectory_recorder.stop_recording()
                    trajectory_recorder.save_trajectory(trajectory_filename)
                    mode_info = "Manual Control"
                    mode_color = (0, 0, 255)  # Blue
                else:
                    trajectory_recorder.start_recording(current_time)
                    mode_info = "Recording"
                    mode_color = (255, 0, 0)  # Red
            
            elif event.key == pygame.K_p:  # P key - Start/Stop Playback
                if trajectory_recorder.is_replaying():
                    trajectory_recorder.stop_replay()
                    mode_info = "Manual Control"
                    mode_color = (0, 0, 255)  # Blue
                else:
                    if trajectory_recorder.load_trajectory(trajectory_filename):
                        trajectory_recorder.start_replay(current_time)
                        mode_info = "Playback"
                        mode_color = (0, 255, 0)  # Green
    
    #Key mappings for debugging
    keys = pygame.key.get_pressed()
    if keys[pygame.K_1]:
        draw_observed_cells = not draw_observed_cells
    elif keys[pygame.K_2]:
        draw_landmark_line = not draw_landmark_line
    elif keys[pygame.K_3]:
        draw_sigma = not draw_sigma
    elif keys[pygame.K_4]:
        draw_estimated_path = not draw_estimated_path
    elif keys[pygame.K_5]:
        draw_sensors = not draw_sensors
    elif keys[pygame.K_6]:
         # x = indices, y = values
        # x = indices, y = values
        x = list(range(len(values)))
        y = values

        # Plotting
        plt.plot(x, y, linestyle='-')
        plt.title('Avg Error')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.plot(x[-1], y[-1], 'o')
        plt.annotate('Final value: '+ str(float.__round__ (y[-1], 4)),
            xy=(x[-1], y[-1]),  # theta, radius
            xytext=(0.65, 0.2),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            )
        plt.grid(True)
        plt.show()
        
    values.append(calculate_mapping_accuracy(grid_probability, walls, obstacles, SCREEN_W, SCREEN_H, GRID_SIZE))
   

    # reset screen
    main_surface.fill((255,255,255))
    if visualize_mapping:
        second_surface.fill((255,255,255))
        
    # re-draw the static map each frame
    for w in walls:
        pygame.draw.rect(main_surface, (0,0,0), w)
    for o in obstacles:
        pygame.draw.rect(main_surface, (0,0,0), o)
    for (i, m_x, m_y) in landmarks:
        pygame.draw.circle(main_surface, "blue", (m_x,m_y), 5)

    # Record trajectory if recording is active
    if trajectory_recorder.is_recording():
        trajectory_recorder.record_point(
            current_time, robot.x, robot.y, robot.theta, robot.v_left, robot.v_right
        )
    
    # Handle trajectory playback if active
    if trajectory_recorder.is_replaying():
        replay_point = trajectory_recorder.get_replay_point(current_time)
        if replay_point:
            x, y, theta, v_left, v_right = replay_point
            # Update robot state directly for playback
            robot.x, robot.y, robot.theta = x, y, theta
            robot.v_left, robot.v_right = v_left, v_right
    
    robot.draw_Robot(main_surface)

    environment_objects = walls + obstacles
    robot.sense(environment_objects, main_surface, draw_sensors, sensor_noise)

    # detect landmarks
    detected_landmarks = robot.detect_landmarks(landmarks, main_surface, draw_landmark_line)
    robot.estimate_pose(kf, landmarks, detected_landmarks, main_surface, position_measurement_noise, theta_mesurement_noise, process_noise)

    # Only process user input if not in playback mode
    if not trajectory_recorder.is_replaying():
        # React on key inputs from the user and adjust wheel speeds
        v_left, v_right = navigate(keys, robot.v_left, robot.v_right)
        robot.v_left = v_left
        robot.v_right = v_right

    # Move the robot and execute collision handling
    robot.move(environment_objects)

    # Add uncertainty ellipse to the robot in intervals of SAMPLE_INTERVAL
    now = pygame.time.get_ticks()
    if now - last_sample_time >= SAMPLE_INTERVAL:
        # grab current estimate
        mean = kf.mu.copy()          # [x, y, θ]
        cov  = kf.sigma[:2, :2].copy()  # only the x–y block
        robot.uncertainty_regions.append((mean, cov))
        last_sample_time = now
    if draw_sigma:
        robot.draw_uncertainty_ellipse(main_surface)
    
    free_cells, occipied_cells = get_observed_cells(robot, GRID_SIZE, grid.shape[0], grid.shape[1])
    
    if draw_observed_cells:
        draw_cells(free_cells, main_surface, GRID_SIZE)
        draw_cells(occipied_cells, main_surface, GRID_SIZE, "green")

    for free in free_cells:
        grid[free[0]][free[1]] += -0.85 / ((kf.sigma[0][0]*10)+1)
    for occ in occipied_cells:
        grid[occ[0]][occ[1]] += 2.2 / ((kf.sigma[0][0]*10)+1)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] > 0:
                draw_cells_filled([(i, j)], main_surface, GRID_SIZE, "purple")

    grid_probability = log_odds_to_prob(grid)
    grid_probability_grey_scale = probs_to_grey_scale(grid_probability)

    goal_x, goal_y = 700, 700
    pygame.draw.circle(main_surface, 'dark cyan', (int(goal_x), int(goal_y)), 7) 
    path = find_path(
        grid_probability,
        (robot.x,robot.y),
        (goal_x,goal_y),
        GRID_SIZE,
        robot_radius=20,
        safety_param=1.2,
        occ_thresh=0.6,
        draw=True,
        surface=main_surface
    )

    if visualize_mapping:
        # draw the grid probabilities 
        for i in range(len(grid_probability_grey_scale)):
            for j in range(len(grid_probability_grey_scale[i])):
                color = grid_probability_grey_scale[i][j]
                pygame.draw.rect(second_surface, (color,color,color), (i*GRID_SIZE, j*GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw the robot's trajectory
    if draw_estimated_path:
        if visualize_mapping:
            robot.drawTrajectories(second_surface)
        else:
            robot.drawTrajectories(main_surface)

    # Draw mode info
    mode_text = info_font.render(mode_info, True, mode_color)
    main_surface.blit(mode_text, (SCREEN_W-180, 25))
    
    # Draw trajectory controls info
    controls_text = info_font.render("R: Record | P: Playback" , True, (0, 0, 0))
    main_surface.blit(controls_text, (SCREEN_W-180, 50))

    # Shows on display
    pygame.display.flip()

    clock.tick(60)

# Before quitting, ensure trajectory is saved if recording
if trajectory_recorder.is_recording():
    trajectory_recorder.stop_recording()
    trajectory_recorder.save_trajectory(trajectory_filename)

pygame.quit()