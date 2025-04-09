from turtle import width
from matplotlib.pylab import rand
import pygame
import math
import numpy as np
from sympy import true
from kinematics import differential_drive_kinematics
import sensors as sn
import maps

# pygame setup
pygame.init()

#set up display 
display_width = 1000
display_height = 1000

# # set up the grid
# num_rows = 20
# num_cols = 20
# row_size = display_height / num_rows
# col_size = display_width / num_cols

screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
dt = 0

# set up the robot's initial position and orientation
# x = col_size / 2 + col_size
# y = row_size / 2 + row_size
theta = 0
x = 100
y = 100
v_left = 0
v_right = 0
r = 40 # robot radius

# maze_string = """
# 11111111111111111111
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 10000000000000000001
# 11111111111111111111
# """

# maze_list = [list(row) for row in maze_string.strip().split("\n")]
# walls = []

max_sensor_range = 100 - r
senors_values = np.full(12, max_sensor_range)

# # make the walls 
# def append_walls():
#     for row in range(num_rows):
#         for col in range(num_cols):
#             if maze_list[row][col] == "1":
#                 x_pos = col * col_size
#                 y_pos = row * row_size
#                 wall_maze = pygame.Rect(x_pos, y_pos, col_size, row_size)
#                 walls.append(wall_maze)

# append_walls()


def draw_robot_text(x, y, r, theta, angle, text):
    my_font = pygame.font.SysFont('Comic Sans MS', 20)
    
    angle = (math.degrees(theta) + angle) % 360
    angle = math.radians(angle)
    
    x2 = x + np.cos(angle) * r
    y2 = y + np.sin(angle) * r
    #adjust from center of textbox coords to topleft of textbox
    x2-=7
    y2-=7

    text_surface = my_font.render(text, False, (0, 0, 0))
    screen.blit(text_surface, (x2,y2))

# check for colliding
def check_x_wall(wall, new_x):
    if new_x < wall.x:
        new_x = wall.x - r
        #print("left")
    elif new_x > wall.x + wall.width:
        new_x = wall.x + wall.width + r
        #print("right")
    return new_x

def check_y_wall(wall, new_y):
    if new_y < wall.y:
        new_y = wall.y - r
        #print("above")
    elif new_y > wall.y + wall.height:
        new_y = wall.y + wall.height + r
        #print("below")
    return new_y



running = True
while running:

    #check for close game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # fill the screen with a color to wipe away anything from last frame
    screen.fill((255,255,255))

    # draw walls  
    # for wall in walls:
    #     #color = list(np.random.choice(range(256), size=3))
    #     pygame.draw.rect(screen, (255,255,255), wall)

    # Define the edges for the map (creates a 980x980 area with a 10 pixel margin).
    edges = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # Define some walls as an example.
    wall_coords = {
        "1": ((0, 200), (700, 200)),
        "2": ((333, 400), (1000, 400)),
        "3": ((333, 400), (333, 800)),
        "4": ((667, 600), (667, 1000)),
    }
    walls = maps.draw_map(screen, edges, wall_coords)

    # calculate new (potential) state
    state = [x, y, theta]
    state_change = differential_drive_kinematics(state, dt, v_left, v_right)
    new_x = x + state_change[0]
    new_y = y + state_change[1]
    new_theta = (theta + state_change[2]) % (2 * math.pi)

    # rectangle representation of the robot
    new_robot_rect = pygame.Rect(new_x - r, new_y - r, 2 * r, 2 * r) 

    colliding_walls = []

    for wall in walls:
        if wall.colliderect(new_robot_rect):
            colliding_walls.append(wall)
    
    if len(colliding_walls) == 1:
            new_x = check_x_wall(colliding_walls[0], new_x)
            new_y = check_y_wall(colliding_walls[0], new_y)
    elif len(colliding_walls) == 2:  
        if abs(colliding_walls[0].x - colliding_walls[1].x) == col_size:
            new_y = check_y_wall(colliding_walls[0],new_x)
        elif abs(colliding_walls[0].y - colliding_walls[1].y) == row_size:
            new_x = check_x_wall(colliding_walls[0], new_y)
    elif len(colliding_walls) > 2:  
        x_coords = np.array([wall.x for wall in colliding_walls])

        # Group rects by unique x values
        unique_x = np.unique(x_coords)
        grouped_by_x = [
            [colliding_walls[i] for i in np.where(x_coords == x)[0]]
            for x in unique_x
        ]

        y_coords = np.array([wall.y for wall in colliding_walls])

        # Group rects by unique y values
        unique_y = np.unique(y_coords)
        grouped_by_y = [
            [colliding_walls[i] for i in np.where(y_coords == y)[0]]
            for y in unique_y
        ]
        
        if len(grouped_by_x) < len(grouped_by_y):
            grouped_by = grouped_by_x
        else:
            grouped_by = grouped_by_y

        grouped_by.sort(key=len,reverse=True)

        if len(grouped_by) == 1:
            if abs(colliding_walls[0].x - colliding_walls[1].x) == col_size:
                new_y = check_y_wall(colliding_walls[0],new_x)
            elif abs(colliding_walls[0].y - colliding_walls[1].y) == row_size:
                new_x = check_x_wall(colliding_walls[0], new_y)
        else:
            for i in range(len(grouped_by)):
                if grouped_by[0][0].x == grouped_by[0][1].x:
                    if y < grouped_by[1][0].y: #above
                        new_y = grouped_by[1][0].y - r
                    else: # below
                        new_y = grouped_by[1][0].y + row_size + r
                    if x < grouped_by[0][0].x: # left
                        new_x = grouped_by[0][0].x - r
                    else: #right
                        new_x = grouped_by[0][0].x + col_size + r
                
                elif grouped_by[0][0].y == grouped_by[0][1].y:
                    if y < grouped_by[0][0].y: #above
                        new_y = grouped_by[0][0].y - r
                    else: # below
                        new_y = grouped_by[0][0].y + row_size + r
                    if x < grouped_by[1][0].x: # left
                        new_x = grouped_by[1][0].x - r
                    else: #right
                        new_x = grouped_by[1][0].x + col_size + r


    # updating state variables
    x, y, theta = new_x, new_y, new_theta
    
    # draw the robot with a direction line
    robot = pygame.draw.circle(screen, "red", (x, y), r)
    pygame.draw.line(screen, (255, 255, 255), (x, y), (np.cos(theta) * r + x, np.sin(theta) * r + y), 2)

    ## draw sensors
    state = (x, y, theta)
    sensor_lines = sn.draw_sensors(screen, state, 100)

    # detect walls
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
    
                
    # react on key inputs from the user and adjust wheel speeds
    # Key events for controlling the robot
    keys = pygame.key.get_pressed()

    # accelerate and decelerate
    if keys[pygame.K_w]:
        if v_left != v_right:
            v_right = max(v_right, v_left)
            v_left = max(v_right, v_left)
        else:
            v_right += 0.2
            v_left += 0.2
    elif keys[pygame.K_s]:
        v_left = max(0, v_left - 0.2)
        v_right = max(0, v_right - 0.2)
    elif keys[pygame.K_d]:
        v_right *= 0.99
    elif keys[pygame.K_a]:
        v_left *= 0.99
    elif keys[pygame.K_q]:
        if v_left == 0 and v_right == 0:
            v_left = 0.3
            v_right = -v_left
    elif keys[pygame.K_e]:
        if v_left == 0 and v_right == 0:
            v_right = 0.3
            v_left = -v_right
    else:
        if v_right < 0 or v_left < 0:
            v_right = v_left = 0
        elif v_right >= v_left:
            v_left = v_right
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

