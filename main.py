import pygame
import math
import numpy as np

from kinematics import differential_drive_kinematics
# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

state = [screen.get_width() /2, screen.get_height()/2, 0]
v_left = 0
v_right = 0

robot_radius = 40

while running:
    #check for close game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill((51,51,51))

    # draw robot
    robot = pygame.draw.circle(screen, "red", [state[0], state[1]], robot_radius)
 
    # draw direction on robot
    pygame.draw.line(screen, (255, 255, 255), [state[0], state[1]], [np.cos(state[2])* robot_radius + state[0], np.sin(state[2])* 40 + state[1]], 5)
    
    #draw wall
    #pygame.draw.line(screen, "white", [500, 900], [500, 200], 10)

    wall = pygame.Rect(750, 80, 300, 100)
    pygame.draw.rect(screen, (255,255,255), wall)


    # Key events
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        v_right += 0.2
        v_left += 0.2
    if keys[pygame.K_s]:
        v_left -= 0.2
        v_right -= 0.2
    if keys[pygame.K_d]:
        v_left = v_left *0.99
    if keys[pygame.K_a]:
        v_right = v_right*0.99
    # Going straight 
    if keys[pygame.K_f]:
        v_right= v_left
    
    
    
    #Update State
    state_change = differential_drive_kinematics(state, dt, v_left , v_right)
    newstate = [state[0] + state_change[0], state[1]+ state_change[1], (state[2] + state_change[2]) % 360]
    
    collide = wall.colliderect(robot)
    if collide:
        if state[0] < wall.x: #and state[1] > wall.y:
            newstate[0] = wall.x - robot_radius
            print("left")

        elif state[0] > wall.x + wall.width: #and state[1] > wall.y:
            newstate[0] = wall.x + wall.width + robot_radius
            print("right")
        elif state[1] < wall.y: #and state[0] > wall.x:
            newstate[1] = wall.y - robot_radius
            print("top")
        elif state[1] > wall.y + wall.height: #and state[0] > wall.x:
            newstate[1] = wall.y + wall.height + robot_radius
            print("bottom")
    
    state = newstate
    
    # Shows on display
    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()