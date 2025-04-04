import pygame
import math
import numpy as np
from kinematics import differential_drive_kinematics
import sensors as sn

# pygame setup
pygame.init()

display_width = 1000
display_height = 500
# set up the grid
num_rows = 10
num_cols = 20
row_size = display_height / num_rows
col_size = display_width / num_cols

screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
dt = 0

# set up the robot's initial position and orientation
x = col_size / 2 + col_size
y = row_size / 2 + row_size
theta = 0

v_left = 0
v_right = 0
r = 40 # robot radius

maze_string = """
11111111111111111111
10000000000000000001
10000000000000000001
10000000000000000001
10000000000000000001
10000000000000000001
10000000000000000001
10000000000000000001
10000000000000000001
11111111111111111111
"""

maze_list = [list(row) for row in maze_string.strip().split("\n")]
walls = []

value = 100 - r
senors_values = np.full(12, value)

def draw_robot_text(x, y, r, theta, angle, text):
    my_font = pygame.font.SysFont('Comic Sans MS', 14)
    
    angle = (math.degrees(theta) + angle) % 360
    angle = math.radians(angle)
    
    x2 = x + np.cos(angle) * r
    y2 = y + np.sin(angle) * r
    #adjust from center of textbox coords to topleft of textbox
    x2-=7
    y2-=7

    text_surface = my_font.render(text, False, (0, 0, 0))
    screen.blit(text_surface, (x2,y2))

    

    

running = True
while running:

    #check for close game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
    # fill the screen with a color to wipe away anything from last frame
    screen.fill((80,80,80))
    
    # draw the walls 
    for row in range(num_rows):
        for col in range(num_cols):
            if maze_list[row][col] == "1":
                x_pos = col * col_size
                y_pos = row * row_size
                wall_maze = pygame.Rect(x_pos, y_pos, col_size, row_size)
                walls.append(wall_maze)
                pygame.draw.rect(screen, (255,255,255), wall_maze)
            elif maze_list[row][col] == "2":
                x_pos = col * col_size
                y_pos = row * row_size
                wall_maze = pygame.Rect(x_pos, y_pos, col_size, row_size)
                pygame.draw.rect(screen, (0,255,0), wall_maze)

    # calculate new (potential) state
    state = [x, y, theta]
    state_change = differential_drive_kinematics(state, dt, v_left, v_right)
    new_x = x + state_change[0]
    new_y = y + state_change[1]
    new_theta = (theta + state_change[2]) % (2 * math.pi)

    new_robot_rect = pygame.Rect(new_x - r, new_y - r, 2 * r, 2 * r) # rectangle representation of the robot

    # check for colliding
    for wall in walls:
        
        if wall.colliderect(new_robot_rect):
            if new_x < wall.x:
                new_x = wall.x - r
            elif new_x > wall.x + wall.width:
                new_x = wall.x + wall.width + r
            elif new_y < wall.y:
                new_y = wall.y - r
            elif new_y > wall.y + wall.height:
                new_y = wall.y + wall.height + r
    
    # updating state variables
    x, y, theta = new_x, new_y, new_theta

    # draw the robot with a direction line
    robot = pygame.draw.circle(screen, "red", (x, y), r)
    pygame.draw.line(screen, (255, 255, 255), (x, y), (np.cos(theta) * r + x, np.sin(theta) * r + y), 5)

    ## draw sensors
    state = (x, y, theta)
    sensor_lines = sn.draw_sensors(screen, state, 100)

    # detect walls
    activatedSensors = []
    for wall in walls:
        new_senors_values =  sn.detect_walls(sensor_lines, wall, state, r)
        for i in range(len(senors_values)):
            if new_senors_values[i] < value:
                senors_values[i] = new_senors_values[i]
                activatedSensors.append(i)
    for i in range(12):
        if i not in activatedSensors:
            senors_values[i] = value
    
                

    # Key events for controlling the robot
    keys = pygame.key.get_pressed()

    # accelerate and decelerate
    if keys[pygame.K_w]:
        v_right = max(v_right, v_left)
        v_left = max(v_right, v_left)
        v_right += 0.2
        v_left += 0.2
    elif keys[pygame.K_s]:
        v_left = max(0, v_left - 0.2)
        v_right = max(0, v_right - 0.2)
    
    elif keys[pygame.K_d]:
        v_left *= 0.99
    elif keys[pygame.K_a]:
        v_right *= 0.99
    elif keys[pygame.K_q]:
        if v_left == 0 and v_right == 0:
            v_left = 0.05
            v_right = -v_left
    elif keys[pygame.K_e]:
        if v_left == 0 and v_right == 0:
            v_right = 0.05
            v_left = -v_right
    # back to straight
    else:
        v_right = v_left
    
    draw_robot_text(x, y, r/3, theta, 270, str(v_left.__round__(1))) # left wheel speed
    draw_robot_text(x, y, r/3, theta, 90, str(v_right.__round__(1))) # right wheel speed
    
    for i in range(12):
        draw_robot_text(x, y, r+10, theta, 30*i, str(round(senors_values[i], 1)))

    # Shows on display
    pygame.display.flip()
    dt = clock.tick(60)

pygame.quit()

