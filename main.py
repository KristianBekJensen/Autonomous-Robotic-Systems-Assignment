import pygame
import math
import numpy as np
from kinematics import differential_drive_kinematics
import sensors as sn

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1600, 900))
clock = pygame.time.Clock()
dt = 0

x, y, theta = screen.get_width() /2, screen.get_height()/2, 0
v_left = 0
v_right = 0
r = 40 # robot radius

def draw_robot_text(x, y, r, theta, angle, text):
    my_font = pygame.font.SysFont('Comic Sans MS', 10)
    
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
    
    # draw a wall
    wall = pygame.Rect(750, 80, 300, 100)
    pygame.draw.rect(screen, (255,255,255), wall)

    # calculate new (potential) state
    state = [x, y, theta]
    state_change = differential_drive_kinematics(state, dt, v_left, v_right)
    new_x = x + state_change[0]
    new_y = y + state_change[1]
    new_theta = (theta + state_change[2]) % (2 * math.pi)

    # check for colliding
    new_robot_rect = pygame.Rect(new_x - r, new_y - r, 2 * r, 2 * r) # rectangle representation of the robot
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
    sn.detect_walls(sensor_lines, wall, state, r)

    # Key events for controlling the robot
    keys = pygame.key.get_pressed()
    # accelerate and decelerate
    if keys[pygame.K_w]:
        v_right += 0.2
        v_left += 0.2
    elif keys[pygame.K_s]:
        v_left = max(0, v_left - 0.2)
        v_right = max(0, v_right - 0.2)
    
    # right and left
    if keys[pygame.K_d]:
        v_left *= 0.99
    elif keys[pygame.K_a]:
        v_right *= 0.99
    # back to striaght
    else:
        v_right = v_left
    
    draw_robot_text(x, y, r/3, theta, 270, str(v_left.__round__(1))) # left wheel speed
    draw_robot_text(x, y, r/3, theta, 90, str(v_right.__round__(1))) # right wheel speed
    
    for i in range(12):
        draw_robot_text(x, y, r+10, theta, 30*i, str(30*i))

    # Shows on display
    pygame.display.flip()
    dt = clock.tick(60)

pygame.quit()

