import math
import numpy as np
import pygame
from leap_ec.problem import ScalarProblem
from fitness import distance_to_target, compute_map_exploration
from mapping import *
from robot import Robot
from kalman_filter import KalmanFilter
from maps import draw_map
 
# We trained two controlers. ExploreControler which is trained with the goal 
# of exploring the map and TargetController which is trained to go directly
# towards the target as soon as target is detected.

class ExploreController:
    """
    A simple neural network with one hidden layer that explores the map by taking sensor distances
    and wheel speeds as inputs and generating new wheel speeds as output
    """
    def __init__(self, genotype, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # convert to np array
        g = np.asarray(genotype, dtype=float)
        idx = 0

        # first layer weights
        end = hidden_size * input_size
        self.W1 = g[idx:idx+end].reshape(hidden_size, input_size)
        idx += end

        # first layer biases
        self.b1 = g[idx:idx+hidden_size]
        idx += hidden_size

        # output layer weights
        end = output_size * hidden_size
        self.W2 = g[idx:idx+end].reshape(output_size, hidden_size)
        idx += end

        # output layer biases
        self.b2 = g[idx:idx+output_size]

    def forward(self, x):
        # forward pass of nn
        h = np.tanh(self.W1.dot(x) + self.b1) # hidden layer activations
        y = np.tanh(self.W2.dot(h) + self.b2) # output in the range of [-1,1]
        return y
    
    def calcInp(self, robot, kf, target_x, target_y, max_dist, d_to_target_from_estimate):
        """ Calculate exploration network input vector for exploration mode. """
        
        inp = np.zeros(self.input_size, dtype=float)

        def chunks(lst, n):
            """ helper function to chunk sensor values into blocks """
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        # take the min sensore distance in each chunk
        sensor_block_values = []
        for chunk in chunks(robot.sensor_values, int(math.log(robot.num_sensors, 6))):
            sensor_block_values.append(min(chunk))
        
        # normalize to [0,1]
        inp[:robot.num_sensors] = np.array(sensor_block_values) / robot.max_sensor_range
        
        # normalize wheel speeds
        inp[robot.num_sensors + 0] = (robot.v_left - robot.min_speed) / (robot.max_speed-robot.min_speed)
        inp[robot.num_sensors + 1] = (robot.v_right - robot.min_speed) / (robot.max_speed-robot.min_speed)
        
        return inp


class TargetController(ExploreController):
    """
    A second one layer nn as controller that in addition to sensors and wheel speeds
    takes the bearing and distance to target as inputs with the goal of reaching the
    target faster. 
    """
    def __init__(self, genotype, input_size, hidden_size, output_size):
        super().__init__(genotype, input_size, hidden_size, output_size)

    def calcInp(self,robot, kf, target_x, target_y, max_dist, d_to_target_from_estimate): 
        """ Calculate target network input vector. """

        # calculate the angle and distance to target from robot's estimated pose.
        dx = target_x - kf.mu[0]
        dy = target_y - kf.mu[1]
        phi = (math.atan2(dy, dx) - robot.theta) % (2*math.pi)
        # normalize bearing into [-pi,pi]
        phi = ((phi + np.pi) % (2 * np.pi)) - np.pi
        
        inp = np.zeros(self.input_size, dtype=float)

        def chunks(lst, n):
            """ helper function to chunk sensor values into blocks """
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        # normalize sensor values
        inp[:robot.num_sensors] = np.array(robot.sensor_values) / robot.max_sensor_range

        # normalize wheel speeds
        inp[robot.num_sensors + 0] = (robot.v_left - robot.min_speed) / (robot.max_speed-robot.min_speed)
        inp[robot.num_sensors + 1] = (robot.v_right - robot.min_speed) / (robot.max_speed-robot.min_speed)
        
        # normalize distance and angle to target
        inp[robot.num_sensors + 2] = d_to_target_from_estimate / max_dist
        inp[robot.num_sensors + 3] = phi / math.pi

        return inp


class MazeSolver(ScalarProblem):
    """
    This class is used to create a problem in evolution.py to be optimized by 
    evolving a phenome (here =genotype vector) that is used to control the robot
    to minimize a fitness function.
    """

    def __init__(
        self,
        maximize,
        visualization,
        num_sensors,
        input_size,
        hidden_size,
        output_size,
        max_steps,
        controller_type,
        fitness_func,
        close_controller=None,
        random=44
    ):
        
        self.visualize_evaluation = visualization
        self.num_sensors = num_sensors
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.random_seed = random
        self.close_controller = close_controller
        self.controller_type = controller_type
        self.fitness_func = fitness_func
        super().__init__(maximize=maximize)

    def evaluate(self, phenome):
        """
        This function is used in evolution.py to run episodes with 
        the given nn weights(=phenome) to calculate fitness score.
        """

        # build the main controller from genotype
        controller = self.controller_type(
            genotype=phenome,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )

        fitness_score = 0.0
        number_runs = 1
        
        for i in range(number_runs):
            """ for each run we create an environment and a robot (similar to main.py) 
            to move in to test the fitness of the robot """

            low_speed_penalty=0
            pygame.init()
            clock = pygame.time.Clock()

            # reset matrices and chose a random target on the map
            collisions = 0
            steps = 0
            target_x = 150 #np.random.uniform(50, 750)
            target_y = 500 #np.random.uniform(50, 750) 
            avg_sigma = 0

            SCREEN_W = SCREEN_H = 800
            WALL_THICKNESS = 4
            GRID_SIZE = WALL_THICKNESS

            # robot's initial pose
            x, y, theta = 100, 100, 0

            min_speed, max_speed = 0.0, 6.0

            # create the robot
            robot = Robot(
                x, y, theta,
                radius=15,
                axel_length=10,
                max_sensor_range=100,
                num_sensors=self.num_sensors,
                min_speed = min_speed,
                max_speed = max_speed
            )
            
            robot.v_left = min_speed
            robot.v_right = min_speed

            process_noise = 0.01
            position_measurement_noise = 0.01
            theta_mesurement_noise = 0.005

            # creating a kalman filter
            kf = KalmanFilter(
                np.array([x, y, theta]),
                np.diag([0.1, 0.1, 0.1]),
                np.diag([process_noise, process_noise, theta_mesurement_noise]),
                np.diag([position_measurement_noise, position_measurement_noise, theta_mesurement_noise])
            )

            # map params
            PAD, NBW, NBH, BS = 20, 8, 8, 100 # padding, number of blocks in width and height, block size
            grid = np.zeros((SCREEN_W//GRID_SIZE, SCREEN_H//GRID_SIZE))
            grid_prob = np.full(grid.shape, 0.5) # starting with prob 0.5 -> unknown grid cells
            
            # pygame window
            flags = 0 if self.visualize_evaluation else pygame.HIDDEN
            screen = pygame.display.set_mode((2*SCREEN_W, SCREEN_H), flags)
            main_surf = screen.subsurface((0,0,SCREEN_W,SCREEN_H))
            # create a second surface in the right side to visualize grid probabilities
            second_surface = screen.subsurface((SCREEN_W, 0, SCREEN_W, SCREEN_H))

            # creating the map
            walls, landmarks, obstacles = draw_map(
                main_surf, num_blocks_w=NBW, num_blocks_h=NBH,
                pad=PAD, 
                wall_thickness=4, 
                n_obstacles=0, # no obstacles, just walls
                random_seed=self.random, # training on different maps
                p_landmark=.3, # helping the robot with beacons at every corner
                wall_h_prob=0.2, # 20% probability of presence of horizontal walls
                wall_v_prob=0.2 # 20% probability of presence of vertical walls
            )

            pygame.display.flip()

            targets_collected = 0
            avg_distance_to_goal = 0.0

        
            # simulation loop
            running = True
            while running:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                steps += 1
                if steps > self.max_steps:
                    break

                pygame.event.pump()
                main_surf.fill((255,255,255))

                # draw walls, obstacles, landmarks, robot, and goal in visualization mode
                if self.visualize_evaluation:
                    for w in walls: pygame.draw.rect(main_surf, (0,0,0), w)
                    for o in obstacles: pygame.draw.rect(main_surf, (0,0,0), o)
                    for _, mx, my in landmarks:
                        pygame.draw.circle(main_surf, 'blue', (mx,my), 5)
                    robot.draw_Robot(main_surf)
                    pygame.draw.circle(main_surf, (255,0,0), (target_x,target_y), 10)

                # sensing objects and update mapping 
                robot.sense(walls+obstacles, main_surf, False)
                detected_landmarks = robot.detect_landmarks(landmarks, main_surf, False)
                free, occ = get_observed_cells(robot, GRID_SIZE, grid.shape[0], grid.shape[1])

                # update occupancy log odds
                for f in free: grid[f] += -0.85/(kf.sigma[0,0]*10+1)
                for o in occ: grid[o] += 2.2/(kf.sigma[0,0]*10+1)
                grid_prob = log_odds_to_prob(grid)

                # use kalman filter to update pose estimate
                robot.estimate_pose(
                    kf,
                    landmarks,
                    detected_landmarks,
                    main_surf,
                    position_measurement_noise,
                    theta_mesurement_noise,
                    process_noise
                )
                
                # define screen diagonal as max dist to use for normalization
                max_dist = math.hypot(SCREEN_W, SCREEN_H)

                distance_to_goal = distance_to_target((robot.x,robot.y),(target_x,target_y))
                avg_distance_to_goal += distance_to_goal
                if distance_to_goal < robot.radius*2:
                    target_x, target_y = np.random.uniform(50, 600), np.random.uniform(50, 199)
                    targets_collected += 1
                d_to_target_from_estimate = distance_to_target((target_x,target_y), (kf.mu[0], kf.mu[1]))
                
                # use the close_controller (=target controller) if you see the target
                if (d_to_target_from_estimate < robot.max_sensor_range) and (self.close_controller is not None):
                    inp = self.close_controller.calcInp(robot, kf, target_x, target_y, max_dist, d_to_target_from_estimate)
                    out = self.close_controller.forward(inp)
                else:
                    inp = controller.calcInp(robot, kf, target_x, target_y, max_dist, d_to_target_from_estimate)
                    out = controller.forward(inp)


                # map back the output of nn for wheel speeds to [min_speed,max_speed] range
                robot.v_left = min_speed + (out[0]+1)/2*(max_speed-min_speed)
                robot.v_right = min_speed + (out[1]+1)/2*(max_speed-min_speed)

                # collision, motion, drawing
                if robot.check_If_Collided(walls+obstacles):
                    collisions += 1
                
                # penalize for not trying to move
                speed = robot.move(walls+obstacles)
                if speed < 4:
                    low_speed_penalty += 5

                # draw the grid probabilities
                if self.visualize_evaluation:
                    grid_probability_grey_scale = probs_to_grey_scale(grid_prob)
                    for i in range(len(grid_probability_grey_scale)):
                        for j in range(len(grid_probability_grey_scale[i])):
                            color = grid_probability_grey_scale[i][j]
                            pygame.draw.rect(second_surface, (color,color,color), (i*GRID_SIZE, j*GRID_SIZE, GRID_SIZE, GRID_SIZE))

                if self.visualize_evaluation:
                    pygame.display.flip()
                    clock.tick(60)

            pygame.quit()

            # after the run calculate how much of the map was explored and distance to goal
            map_unexplored = compute_map_exploration(grid_prob, threshold=1e-3)
            avg_distance_to_goal = avg_distance_to_goal / steps

            # final score
            fitness_score += self.fitness_func(
                num_collisions=collisions,
                num_time_steps=steps,
                dist_to_target=avg_distance_to_goal,
                map_unexplored=map_unexplored,
                speed=low_speed_penalty,
                targets_collected=targets_collected
            )
        # return avg fitness and avg sigma    
        return fitness_score/number_runs, avg_sigma/steps/number_runs
