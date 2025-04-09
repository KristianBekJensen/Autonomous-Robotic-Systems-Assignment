import pygame
import math
import numpy as np
from kinematics import differential_drive_kinematics
from navigate import navigate
import sensors as sn
import maps

# pygame setup
pygame.init()

debug = True

#set up display 
display_width = 1000
display_height = 1000

screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
dt = 0
r = 40 # robot radius

# Setup Robot intial Pos
theta = 0
x = 100
y = 100
v_left = 0
v_right = 0

max_sensor_range = 100 - r
senors_values = np.full(12, max_sensor_range)

def point_on_circles_circumference(x, y, r, theta, angle):
    angle = (math.degrees(theta) + angle) % 360
    angle = math.radians(angle)
    
    x2 = x + np.cos(angle) * r
    y2 = y + np.sin(angle) * r

    return x2, y2

def draw_robot_text(x, y, r, theta, angle, text):
    my_font = pygame.font.SysFont('Comic Sans MS', 14)
    
    x2, y2 = point_on_circles_circumference(x, y, r, theta, angle)
    
    # Adjust coords from center of textbox to coords topleft of textbox
    x2-=7
    y2-=7

    text_surface = my_font.render(text, False, (0, 0, 0))
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
    edges = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # Define some walls as an example.
    wall_coords = {
        "1": ((0, 200), (700, 200)),
        "2": ((333, 400), (1000, 400)),
        "3": ((333, 400), (333, 800)),
        "4": ((667, 600), (667, 1000)),
    }
    wall_thickness = 3
    walls = maps.draw_map(screen, edges, wall_coords, wall_thickness)

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
    
    # Draw the robot and it's direction line
    pygame.draw.circle(screen, "red", (x, y), r) # Draw robot
    pygame.draw.line(screen, (255, 255, 255), (x, y), (np.cos(theta) * r + x, np.sin(theta) * r + y), 2)

    ## draw sensors
    state = (x, y, theta)
    sensor_lines = sn.draw_sensors(screen, state, 100, debug)

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
    
    for i in range(12): # draw sensor values
        draw_robot_text(x, y, r+10, theta, 30*i, str(round(senors_values[i], 1)))

    # Shows on display
    pygame.display.flip()
    dt = clock.tick(60)

pygame.quit()

