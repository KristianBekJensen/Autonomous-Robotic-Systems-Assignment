import math
import numpy as np
import pygame

from leap_ec.problem import ScalarProblem
from Fitness import distance_to_target, compute_map_exploration
from mapping import *
from path_finder import find_path
from robot import Robot
from kalman_filter import KalmanFilter
from maps import draw_map

class ExploreController:
    def __init__(self, genotype, input_size, hidden_size, output_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        g = np.asarray(genotype, dtype=float)
        idx = 0

        end = hidden_size * input_size
        self.W1 = g[idx:idx+end].reshape(hidden_size, input_size)
        idx += end

        self.b1 = g[idx:idx+hidden_size]
        idx += hidden_size

        end = output_size * hidden_size
        self.W2 = g[idx:idx+end].reshape(output_size, hidden_size)
        idx += end

        self.b2 = g[idx:idx+output_size]
        # idx += output_size

    def forward(self, x):
        # x: (input_size,)
        h = np.tanh(self.W1.dot(x) + self.b1)    # hidden activations
        y = np.tanh(self.W2.dot(h) + self.b2)    # output in [-1,1]
        return y
    
    def calcInp(self, robot, target_x, target_y, d_to_target_from_estimate):
        # --- NEW: controller step ---
        # build input vector ∈ ℝ^input_size
        inp = np.zeros(self.input_size, dtype=float)
        # normalize sensors to [0,1]
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        sensor_block_values = []
        for chunk in chunks(robot.sensor_values, int(math.log(robot.num_sensors, 6))):
            sensor_block_values.append(min(chunk))
        inp[:robot.num_sensors] = np.array(sensor_block_values) / robot.max_sensor_range
        # normalize wheel speeds
        inp[robot.num_sensors + 0] = (robot.v_left  - robot.min_speed)/(robot.max_speed-robot.min_speed)
        inp[robot.num_sensors + 1] = (robot.v_right - robot.min_speed)/(robot.max_speed-robot.min_speed)
        
        return inp

class TargetController(ExploreController):
    def __init__(self, genotype, input_size, hidden_size, output_size):
        super().__init__(genotype, input_size, hidden_size, output_size)

    # Overrides
    def calcInp(self, robot, target_x, target_y, max_dist, d_to_target_from_estimate): 
        # calculate the distance to target
        # angle to target
        dx = target_x - robot.x
        dy = target_y - robot.y
        phi = (math.atan2(dy, dx) - robot.theta) % (2*math.pi)
        
        inp = np.zeros(self.input_size, dtype=float)
        # normalize sensors to [0,1]
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        sensor_block_values = []
        for chunk in chunks(robot.sensor_values, int(math.log(robot.num_sensors, 6))):
            sensor_block_values.append(chunk[0]) # Retrain if want to use proper Chunks
        inp[:robot.num_sensors] = np.array(sensor_block_values) / robot.max_sensor_range
        # normalize wheel speeds
        inp[robot.num_sensors + 0] = (robot.v_left  - robot.min_speed)/(robot.max_speed-robot.min_speed)
        inp[robot.num_sensors + 1] = (robot.v_right - robot.min_speed)/(robot.max_speed-robot.min_speed)
        # normalize ang
        # distance and angle to target from our estimated position 
        inp[robot.num_sensors + 2] = d_to_target_from_estimate / max_dist
        inp[robot.num_sensors + 3] = phi / math.pi
        return inp


class MazeSolver(ScalarProblem):
    """
    'phenome' is a real vector of NN weights used to control the robot.
    """

    def __init__(self,
                 maximize: bool,
                 visualization: bool,
                 num_sensors: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 max_steps: int,
                 controller_type,
                 fitness_func,
                 close_controller = None,
                 random = 44):
        
        self.visualize_evaluation = visualization
        self.num_sensors = num_sensors
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.random = random
        self.close_controller = close_controller
        self.controller_type = controller_type
        self.fitness_func = fitness_func
        super().__init__(maximize=maximize)

    def evaluate(self, phenome):
        # # Build controller from genotype
        controller = self.controller_type(
            genotype=phenome,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )

        fitness_score = 0.0
        number_runs = 2
        
        for i in range(number_runs):

            total_speed=0
            # --- Pygame & robot setup (same as before) ---
            pygame.init()
            clock = pygame.time.Clock()

            # Metrics
            collisions = 0
            steps      = 0
            #max_steps  = float('inf') if self.visualize_evaluation else 400
            #target_x, target_y = 500, 500
            # make the target random
            target_x = np.random.uniform(50, 750)
            target_y = np.random.uniform(50, 750) 
            avg_sigma = 0

            # Robot initial pose
            SCREEN_W = SCREEN_H = 800
            WALL_THICKNESS = 4
            GRID_SIZE = WALL_THICKNESS

            #x = np.random.uniform(50, SCREEN_W-50)
            #y = np.random.uniform(50, SCREEN_H-50)
            #theta = np.random.uniform(-np.pi, np.pi)
            x, y, theta = 100, 100, 0

            min_speed, max_speed = 0.0, 6.0
            robot = Robot(x, y, theta,
                        radius=15,
                        axel_length=10,
                        max_sensor_range=100,
                        num_sensors=self.num_sensors,
                        min_speed = min_speed,
                        max_speed = max_speed)
            robot.v_left  = min_speed
            robot.v_right = min_speed

            process_noise              = 0.01
            position_measurement_noise = 0.01
            theta_mesurement_noise     = 0.005

            # Kalman filter & map
            kf = KalmanFilter(
                np.array([x, y, theta]),                     # init_state
                np.diag([0.1, 0.1, 0.1]),                    # init_covariance
                np.diag([process_noise, process_noise,       # process_noise R
                        theta_mesurement_noise]),
                np.diag([position_measurement_noise,         # measurement_noise Q
                        position_measurement_noise,
                        theta_mesurement_noise])
            )
            #SCREEN_W = SCREEN_H = 800
            PAD, NBW, NBH, BS = 20, 8, 8, 100
            grid = np.zeros((SCREEN_W//4, SCREEN_H//4))
            grid_prob = np.full(grid.shape, 0.5)

            # Pygame window
            flags = 0 if self.visualize_evaluation else pygame.HIDDEN
            screen = pygame.display.set_mode((2*SCREEN_W, SCREEN_H), flags)
            main_surf = screen.subsurface((0,0,SCREEN_W,SCREEN_H))
            second_surface = screen.subsurface((SCREEN_W, 0, SCREEN_W, SCREEN_H))

            walls, landmarks, obstacles = draw_map(
                main_surf, num_blocks_w=NBW, num_blocks_h=NBH,
                pad=PAD, wall_thickness=4, n_obstacles=0, random_seed=self.random, p_landmark=1.0, wall_h_prob=0.2, wall_v_prob=0.2
            )

            pygame.display.flip()

            targets_collected = 0
            distance_to_goal = 0.0

        
            # --- Simulation loop ---
            running = True
            while running:
                # catch the window-close event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                steps += 1
                if steps > self.max_steps:
                    break

                """ if steps % 100 == 0:
                    target_x = np.random.uniform(50, 750)
                    target_y = np.random.uniform(50, 750) """

                pygame.event.pump()
                main_surf.fill((255,255,255))

                if self.visualize_evaluation:
                    for w in walls:       pygame.draw.rect(main_surf, (0,0,0), w)
                    for o in obstacles:   pygame.draw.rect(main_surf, (0,0,0), o)
                    for _, mx, my in landmarks:
                        pygame.draw.circle(main_surf, 'blue', (mx,my), 5)
                    robot.draw_Robot(main_surf)
                    pygame.draw.circle(main_surf, (255,0,0), (target_x,target_y), 10)

                # Sense & localize
                robot.sense(walls+obstacles, main_surf, False)
                detected_landmarks = robot.detect_landmarks(landmarks, main_surf, False)
                free, occ = get_observed_cells(
                    robot,
                    GRID_SIZE,                 # grid cell size
                    grid.shape[0],      # number of columns
                    grid.shape[1]       # number of rows
                )
                for f in free: grid[f] += -0.85/(kf.sigma[0,0]*10+1)
                for o in occ: grid[o] +=  2.2/(kf.sigma[0,0]*10+1)
                grid_prob = log_odds_to_prob(grid)

                robot.estimate_pose(
                    kf,
                    landmarks,
                    detected_landmarks,
                    main_surf,
                    position_measurement_noise,  # ← matches 5th param
                    theta_mesurement_noise,      # ← matches 6th param
                    process_noise                # ← matches 7th param
                )
                

                if distance_to_target((target_x,target_y), (robot.x, robot.y)) < 30:
                    target_x, target_y = np.random.randint(0+PAD, SCREEN_W-PAD), np.random.randint(0+PAD, SCREEN_H-PAD)
                    targets_collected += 1
                
                
                max_dist = math.hypot(SCREEN_W, SCREEN_H)  # normalize max distance
                d_to_target_from_estimate = distance_to_target((target_x,target_y), (kf.mu[0], kf.mu[1]))
                if (d_to_target_from_estimate < robot.max_sensor_range) and (self.close_controller is not None):
                    inp = self.close_controller.calcInp(robot, target_x, target_y, max_dist, d_to_target_from_estimate)
                    out = self.close_controller.forward(inp)   # 2 outputs in [−1,1]
                else:
                    inp = controller.calcInp(robot, target_x, target_y, max_dist, d_to_target_from_estimate)
                    out = controller.forward(inp)   # 2 outputs in [−1,1]


                # map back to [min_speed,max_speed]
                robot.v_left  = min_speed + (out[0]+1)/2*(max_speed-min_speed)
                robot.v_right = min_speed + (out[1]+1)/2*(max_speed-min_speed)

                # collision, motion, drawing
                if robot.check_If_Collided(walls+obstacles):
                    collisions += 1
                distance_to_goal += distance_to_target((robot.x,robot.y),(target_x,target_y))
                if distance_to_goal < robot.radius:
                    break

                speed = robot.move(walls+obstacles)
                if speed < 4:
                    total_speed += 5

                if self.visualize_evaluation:
                    grid_probability_grey_scale = probs_to_grey_scale(grid_prob)
                    # draw the grid probabilities 
                    for i in range(len(grid_probability_grey_scale)):
                        for j in range(len(grid_probability_grey_scale[i])):
                            color = grid_probability_grey_scale[i][j]
                            pygame.draw.rect(second_surface, (color,color,color), (i*GRID_SIZE, j*GRID_SIZE, GRID_SIZE, GRID_SIZE))

                if self.visualize_evaluation:
                    pygame.display.flip()
                    clock.tick(60)

            # shutdown
            pygame.quit()

            map_unexplored = compute_map_exploration(grid_prob, threshold=1e-3)
            distance_to_goal = distance_to_goal / steps

            # final score
            fitness_score += self.fitness_func(
                num_collisions=collisions,
                num_time_steps=steps,
                dist_to_target=distance_to_goal,
                map_unexplored=map_unexplored,
                speed = total_speed,
                targets_collected = targets_collected
            )
        return fitness_score / number_runs,  avg_sigma/steps/number_runs
