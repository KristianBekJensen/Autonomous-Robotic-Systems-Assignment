from doctest import debug
import numpy as np
import pygame
import math
import itertools

from pyparsing import col
from kinematics import differential_drive_kinematics
from landmark import two_point_triangulate, get_landmark_measurements
from utils import draw_covariance_ellipse

class Robot:
    
    def __init__(self, 
                 x: float = 100.0, 
                 y: float = 100.0, 
                 theta: float = 0.0, 
                 radius: float = 20.0,
                 axel_length: float = 10.0,
                 max_sensor_range: float = 100.0,
                 num_sensors: int = 12,
                ):
        
        """Initialize the robot with given parameters."""
        # State
        self.x = x
        self.y = y
        self.theta = theta  # radians
        self.true_pose = (x, y, theta)
        self.estimated_pose = (x, y, theta)
        
        # Physical attributes
        self.radius = radius
        self.axel_length = axel_length
        
        # Motion attributes
        self.v_left = 0.0
        self.v_right = 0.0
        
        # Sensors
        self.num_sensors = num_sensors
        self.max_sensor_range = max_sensor_range
        self.sensor_lines = []
        
        # Trajectories
        self.true_poses = [(x, y, theta)]
        self.estimated_poses = [(x, y, theta)]
        self.sensor_values = np.full(self.num_sensors, max_sensor_range) 
        self.uncertainty_regions = []
        
    def move(self, walls):
        
        state_change = differential_drive_kinematics(
            [self.x, self.y, self.theta], 
            self.v_left, 
            self.v_right,
            self.axel_length
        )

        new_x = self.x + state_change[0]
        new_y = self.y + state_change[1]
        new_theta = (self.theta + state_change[2]) % (2 * math.pi)

        new_x, new_y, _ = self.check_collisions(new_x, new_y, walls)
        
        distance_moved = math.sqrt(abs(self.x - new_x)**2 + (self.y - new_y)**2)

        # Update state variables after collision handling
        self.x, self.y, self.theta = new_x, new_y, new_theta
        robot_pose = np.array([self.x, self.y, self.theta])
        self.true_poses.append(robot_pose)
        
        return distance_moved

    def check_If_Collided(self, walls):
        state_change = differential_drive_kinematics(
            [self.x, self.y, self.theta], 
            self.v_left, 
            self.v_right,
            self.axel_length
        )

        new_x = self.x + state_change[0]
        new_y = self.y + state_change[1]

        _, _, collided = self.check_collisions(new_x, new_y, walls)
        return collided 

    def check_collisions(self, new_x, new_y, walls):
        """Check and resolve collisions with multiple walls"""
        # Future possible position
        test_x, test_y = new_x, new_y
        collided = False
        
        # Check each wall for collision
        collided_walls = []
        for wall in walls:

            # Getclosest point on rectangle to robot center
            closest_x = max(wall.x, min(test_x, wall.x + wall.width))
            closest_y = max(wall.y, min(test_y, wall.y + wall.height))
            
            # Calculate distance between closest point and circle center
            distance_x = test_x - closest_x
            distance_y = test_y - closest_y
            distance = math.sqrt(distance_x**2 + distance_y**2)
            
            # Check if colliding
            if distance < self.radius:
                collided_walls.append((wall, closest_x, closest_y, distance))
                collided = True

        # If no collided walls, move to the possible position
        if not collided_walls:
            return test_x, test_y, collided
        
        # Handle collisions and adjust the new position (x,y adjusted)
        adjusted_x, adjusted_y = self.resolve_collisions(test_x, test_y, collided_walls)
        return adjusted_x, adjusted_y, collided
    
    def resolve_collisions(self, test_x, test_y, collided_walls):
        """Resolve collisions with potentially multiple walls"""

        # Initialize with  attempted position
        adjusted_x, adjusted_y = test_x, test_y
        
        for wall, closest_x, closest_y, distance in collided_walls:
            # Calculate normal vector from closest point of wall to the robot center and calculate push direction

            if distance > 0:  # Avoid division by zero
                normal_x = (test_x - closest_x) / distance
                normal_y = (test_y - closest_y) / distance
            else:
                # If center is exactly on edge, use direction between previous and attempted position
                dx = test_x - self.x
                dy = test_y - self.y
                magnitude = math.sqrt(dx**2 + dy**2)
                if magnitude > 0:
                    normal_x = dx / magnitude
                    normal_y = dy / magnitude
                else:
                    # Default push direction if no movement
                    normal_x, normal_y = 0, -1
            
            # Calculate how much we need to push the circle out
            push_distance = self.radius - distance
            if push_distance > 0:
                # Push the circle out along the normal vector
                adjusted_x += normal_x * push_distance
                adjusted_y += normal_y * push_distance
        
        return adjusted_x, adjusted_y
  
    def detect_landmarks(self, landmarks, screen, draw):
        detected_landmarks = []
        for landmark in landmarks: 
            i, m_x, m_y = landmark
            distance = math.sqrt((self.x - m_x) ** 2 + (self.y - m_y) ** 2)

            if distance < self.max_sensor_range:
                if draw:
                    pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (m_x, m_y), 2)
                detected_landmarks.append(landmark)

        return detected_landmarks
    
    def estimate_pose(self, kf, landmarks, detected_landmarks, screen, position_measurement_noise, theta_mesurement_noise, process_noise):
        
        state = [self.x, self.y, self.theta]
        state_change = differential_drive_kinematics(state, self.v_left, self.v_right, self.axel_length)

        # Predict step
        v = (self.v_left + self.v_right) / 2
        omega = state_change[2]
        u = np.array([v, omega])
        estimated_pose, sigma = kf.predict(u, 1, process_noise)

        # Update step
        measurements = get_landmark_measurements(detected_landmarks, (self.x, self.y,self.theta))
        num_detected_landmarks = len(measurements)
        if num_detected_landmarks == 2:
            p1, p2, estimate_theta = two_point_triangulate(measurements, landmarks, (self.x, self.y, self.theta), 
                                                           position_measurement_noise, 
                                                           theta_mesurement_noise)
            if p1 != (0,0):
                
                z = [p1[0], p1[1], estimate_theta]
                estimated_pose, sigma = kf.update(z)
        # more than 2 detected landmarks
        elif num_detected_landmarks > 2:
            zs = []
            avg_theta = 0
            # for every unique pair of landmark measurements
            for i, j in itertools.combinations(range(num_detected_landmarks), 2):
                meas_pair = [measurements[i], measurements[j]]
                p1, p2, estimate_theta = two_point_triangulate(meas_pair, landmarks, (self.x, self.y, self.theta), position_measurement_noise, theta_mesurement_noise)
                
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
        
        self.estimated_pose = estimated_pose 
        self.estimated_poses.append(estimated_pose)

        return estimated_pose
    
    def draw_Robot(self, screen):
        # Draw the robot and its direction line
        pygame.draw.circle(screen, "black", (self.x, self.y), self.radius)  # Draw robot
        pygame.draw.circle(screen, "white", (self.x, self.y), self.radius - 2)  # Draw robot
        pygame.draw.line(screen, (140, 0, 140), 
                         (self.x, self.y), 
                         (np.cos(self.theta) * self.radius + self.x, 
                          np.sin(self.theta) * self.radius + self.y), 3)
        self.draw_robot_text(screen, self.x, self.y, self.radius/3, self.theta, 270, str(self.v_left.__round__(1))) # draw left wheel speed
        self.draw_robot_text(screen, self.x, self.y, self.radius/3, self.theta, 90, str(self.v_right.__round__(1))) # draw right wheel speed
        
    def drawTrajectories(self, screen):
        if len(self.true_poses) > 1:
            pts = [(px, py) for px, py, _ in self.true_poses]
            pygame.draw.lines(screen, (0, 150, 150), False, pts, 2)
        if len(self.estimated_poses) > 1:
            pts = [(px, py) for px, py, _ in self.estimated_poses]
            pygame.draw.lines(screen, "red", False, pts, 2)

    def draw_robot_text(self, screen, x, y, r, theta, angle, text):
        
        font = pygame.font.SysFont('Arial', 10)

        def point_on_circles_circumference(x, y, r, theta, angle):
            angle = (math.degrees(theta) + angle) % 360
            angle = math.radians(angle)
            
            x2 = x + np.cos(angle) * r
            y2 = y + np.sin(angle) * r

            return x2, y2
        
        x2, y2 = point_on_circles_circumference(x, y, r, theta, angle)
        
        # Adjust coords from center of textbox to coords topleft of textbox
        x2-=7
        y2-=7

        text_surface = font.render(text, False, (0, 0, 0))
        screen.blit(text_surface, (x2,y2))
    
    def draw_uncertainty_ellipse(self, screen):
        for mean, cov in self.uncertainty_regions:
            draw_covariance_ellipse(
                screen,
                mean=(mean[0], mean[1]),
                cov=cov,
                n_std=4.0,           
                num_points=64,
                color="orange",
                width=2
            )
    
    def sense(self, walls, screen=None, draw=False, sensor_noise=0):
        self.sensor_values = np.full(self.num_sensors, self.max_sensor_range, dtype=float)
        self.sensor_lines = []

        for i in range(self.num_sensors):
            angle = self.theta + i * (2 * math.pi / self.num_sensors)
            start = (self.x, self.y)
            max_end = (self.x + (self.max_sensor_range + self.radius) * math.cos(angle), self.y + (self.max_sensor_range + self.radius) * math.sin(angle))

            # find nearest wall intersection
            min_dist = self.max_sensor_range
            hit_point = None
            for wall in walls:
                clipped = wall.clipline(start, max_end)
                if clipped:
                    (hx, hy), _ = clipped
                    d = math.hypot(hx - self.x, hy - self.y) - self.radius
                    if 0 <= d < min_dist:
                        min_dist = d
                        hit_point = (hx, hy)

            self.sensor_values[i] = min_dist + np.random.normal(0, sensor_noise)

            # choose draw end
            draw_end = hit_point if hit_point is not None else max_end
            self.sensor_lines.append((i, start, draw_end))
            if draw and screen:
                pygame.draw.line(screen, (0, 0, 0), start, draw_end, 1)

        return self.sensor_values
