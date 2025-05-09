from matplotlib.pylab import number
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
from fitness import distance_to_target, fitness
from leap_ec.problem import ScalarProblem

class MazeSolver(ScalarProblem):

    def __init__(self):
        super().__init__(maximize=True)

    
    def run(self, phenome):
        # pygame setup
        pygame.init()

        default_font = pygame.font.SysFont('Arial', 10)
        info_font = pygame.font.SysFont('Arial', 18)

        # Evaluation metrics 
        number_collisions = 0
        number_time_steps = 0
        max_time_steps = 10000
        distance_to_target_value = 0
        target_position = [500, 500]


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
            obstacle_color=(0,0,0),
            random_seed=42
        )



        pygame.display.flip()

        values = []
        # Game Loop 
        running = True
        while running:
            current_time = pygame.time.get_ticks()

            number_time_steps += 1

            if number_time_steps > max_time_steps:
                running = False


            # reset screen
            main_surface.fill((255,255,255))

                
            """ # re-draw the static map each frame
            for w in walls:
                pygame.draw.rect(main_surface, (0,0,0), w)
            for o in obstacles:
                pygame.draw.rect(main_surface, (0,0,0), o)
            for (i, m_x, m_y) in landmarks:
                pygame.draw.circle(main_surface, "blue", (m_x,m_y), 5) 
            
            robot.draw_Robot(main_surface)
            """

            environment_objects = walls + obstacles
            robot.sense(environment_objects, main_surface, False, sensor_noise)

            # detect landmarks
            detected_landmarks = robot.detect_landmarks(landmarks, main_surface, False)
            robot.estimate_pose(kf, landmarks, detected_landmarks, main_surface, position_measurement_noise, theta_mesurement_noise, process_noise)

            

            if robot.check_If_Collided(environment_objects):
                number_collisions += 1
            print("Collisions: ", number_collisions)
            print("Time Steps: ", number_time_steps)
            distance_to_target_value = distance_to_target([robot.x, robot.y], target_position)
            print("Distance to Target Value: ", distance_to_target_value)

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
            
            free_cells, occipied_cells = get_observed_cells(robot, GRID_SIZE, grid.shape[0], grid.shape[1])
            
            for free in free_cells:
                grid[free[0]][free[1]] += -0.85 / ((kf.sigma[0][0]*10)+1)
            for occ in occipied_cells:
                grid[occ[0]][occ[1]] += 2.2 / ((kf.sigma[0][0]*10)+1)


            grid_probability = log_odds_to_prob(grid)
            grid_probability_grey_scale = probs_to_grey_scale(grid_probability)

            


            # Shows on display
            pygame.display.flip()

            clock.tick(60)

        

        pygame.quit()

        collison_weight = 1
        time_weight = 1
        distance_weight = 1

        return fitness(number_collisions, number_time_steps, distance_to_target_value, collison_weight, time_weight, distance_weight)

    def evaluate(self, phenome):

        return self.run(phenome)