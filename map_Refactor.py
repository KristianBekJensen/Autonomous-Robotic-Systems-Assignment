import pygame
import random

# Colors
BACKGROUND_COLOR = (255, 255, 255)
WALL_COLOR = (0, 0, 0)
LANDMARK_COLOR = (0, 0, 255)

# Each block contains a [top, right, bottom, left] wall flag
def generate_sample_map(n_x, n_y):
    grid = []
    for y in range(n_y):
        row = []
        for x in range(n_x):
            # 20% chance for each wall to be True (i.e., present)
            walls = [random.random() < 0.2 for _ in range(4)]
            row.append(walls)
        grid.append(row)
    return grid

def draw_map(screen, grid, wall_thickness, block_width, block_height):
    wall_list = []
    landmark_list = []
    landmark_id = 0
    for y, row in enumerate(grid):
        for x, walls in enumerate(row):
            px = x * block_width
            py = y * block_height
            
            if walls[0]:  # Top wall
                rect = pygame.Rect(
                    px, py,
                    block_width , wall_thickness
                )
                pygame.draw.rect(screen, WALL_COLOR, rect)
                wall_list.append(rect)

            if walls[1]:  # Right wall
                rect = pygame.Rect(
                    px + block_width - wall_thickness, py,
                    wall_thickness, block_height
                )
                pygame.draw.rect(screen, WALL_COLOR, rect)
                wall_list.append(rect)


            if walls[2]:  # Bottom wall
                rect = pygame.Rect(
                    px, py + block_height - wall_thickness,
                    block_width, wall_thickness
                )
                pygame.draw.rect(screen, WALL_COLOR, rect)
                wall_list.append(rect)


            if walls[3]:  # Left wall
                rect = pygame.Rect(
                    px, py,
                    wall_thickness, block_height
                )
                pygame.draw.rect(screen, WALL_COLOR, rect)
                wall_list.append(rect)

            # Check for landmarks

            # Top-Right Corner
            if walls[0] and walls[1]:
                landmark = (landmark_id, px + block_width, py)
                landmark_list.append(landmark)
                landmark_id += 1

            # Bottom-Right Corner
            if walls[1] and walls[2]:
                landmark = (landmark_id, px + block_width, py + block_height)
                landmark_list.append(landmark)
                landmark_id += 1

            # Bottom-Left Corner
            if walls[2] and walls[3]:
                landmark = (landmark_id, px, py + block_height)
                landmark_list.append(landmark)
                landmark_id += 1

            # Top-Left Corner
            if walls[3] and walls[0]:
                landmark = (landmark_id, px, py)
                landmark_list.append(landmark)
                landmark_id += 1        
    
    return wall_list, landmark_list

def check_x_wall(wall, new_x, x, r):
    if x <= wall.x: # Robot left
        new_x = wall.x - r
    elif x >= wall.x + wall.width: # Robot right
        new_x = wall.x + wall.width + r
    return new_x

def check_y_wall(wall, new_y, y, r):
    if y <= wall.y: # Robot above
        new_y = wall.y - r
    elif y >= wall.y + wall.height: # Robot below
        new_y = wall.y + wall.height + r
    return new_y

def draw_cells(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect(
            cell[1] * block_width,
            cell[0] * block_width,
            block_width,
            block_width
        )
        pygame.draw.rect(screen, color, rect, 1)  # Draw the cell with a red border

def draw_cells_filled(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect(
            cell[1] * block_width,
            cell[0] * block_width,
            block_width,
            block_width
        )
        pygame.draw.rect(screen, color, rect)  # Draw the cell with a red border