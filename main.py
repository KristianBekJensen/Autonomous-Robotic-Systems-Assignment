# Example file showing a circle moving on screen
import pygame
import math

from ky import differential_drive_kinematics
# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0
text_color = (255, 255, 255)
font = pygame.font.Font(pygame.font.match_font('corbel', True), 20)


state = [screen.get_width() /2, screen.get_height()/2, 0]
v_left = 0
v_right = 0

def draw_orientation(robot_state):
        x, y, theta = robot_state

        center = (700, 80)  
        radius = 50
        pygame.draw.circle(screen, text_color, center, radius, 2) 

        for angle in range(0, 360, 45):
            spoke_angle = math.radians(angle)
            spoke_x = center[0] + radius * math.cos(spoke_angle)
            spoke_y = center[1] + radius * math.sin(spoke_angle)
            pygame.draw.line(screen, text_color, center, (spoke_x, spoke_y), 1)

        theta_angle = theta % (2 * math.pi)
        bold_spoke_x = center[0] + radius * math.cos(theta_angle)
        bold_spoke_y = center[1] + radius * math.sin(theta_angle)
        pygame.draw.line(screen, text_color, center, (bold_spoke_x, bold_spoke_y), 3)  

        orientation_text = font.render("Orientation", True, text_color)
        screen.blit(orientation_text, (center[0] - radius, center[1] + radius + 20))

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")

    pygame.draw.circle(screen, "red", [state[0], state[1]], 40)
 

    pygame.draw.line(screen,"yellow", [200,400], [400,400], 5)
    
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

    s = differential_drive_kinematics(state, dt,v_left , v_right)
    state[2] = (state[2] + s[2]) % 360
    state = [state[0] + s[0], state[1]+ s[1], (state[2] + s[2]) % 360]

    draw_orientation(state)
    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()