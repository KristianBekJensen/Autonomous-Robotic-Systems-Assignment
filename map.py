import pygame
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def draw_map(screen, horiz, vert, pad=20, grid_w=16, grid_h=9, wall_color=(0, 0, 0), wall_thickness=1):
    screen_w, screen_h = screen.get_size()
    map_w = screen_w - 2 * pad
    map_h = screen_h - 2 * pad
    cell_w = map_w / grid_w
    cell_h = map_h / grid_h

    wall_list = []

    # Horizontal walls
    for r in range(grid_h + 1):
        y = pad + r * cell_h
        for c, present in enumerate(horiz[r]):
            if present:
                x = pad + c * cell_w
                rect = pygame.Rect(int(x), int(y), int(cell_w), wall_thickness)
                pygame.draw.rect(screen, wall_color, rect)
                wall_list.append(rect)

    # Vertical walls
    for r in range(grid_h):
        y = pad + r * cell_h
        for c, present in enumerate(vert[r]):
            if present:
                x = pad + c * cell_w
                rect = pygame.Rect(int(x), int(y), wall_thickness, int(cell_h))
                pygame.draw.rect(screen, wall_color, rect)
                wall_list.append(rect)

    return wall_list

def compute_landmarks(horiz, vert, screen, pad=20, grid_w=16, grid_h=9, p_landmark=0.25):
    sw, sh = screen.get_size()
    map_w = sw - 2*pad
    map_h = sh - 2*pad
    cell_w = map_w / grid_w
    cell_h = map_h / grid_h

    landmarks = []
    l_id = 0

    # Horizontal segments: r in [0..grid_h], c in [0..grid_w-1]
    for r in range(grid_h + 1):
        y = pad + r * cell_h
        for c, present in enumerate(horiz[r]):
            if not present:
                continue

            # endpoints in cell‐coords: (c, r) and (c+1, r)
            x1, y1 = pad + c * cell_w, y
            x2, y2 = pad + (c+1) * cell_w, y

            if random.random() < p_landmark:
                landmarks.append((l_id, int(x1), int(y1)))
                l_id += 1
            if random.random() < p_landmark:
                landmarks.append((l_id, int(x2), int(y2)))
                l_id += 1

    # Vertical segments: r in [0..grid_h-1], c in [0..grid_w]
    for r in range(grid_h):
        x = pad + c * cell_w
        for c, present in enumerate(vert[r]):
            if not present:
                continue

            # endpoints: (c, r) and (c, r+1)
            x = pad + c * cell_w
            y1 = pad + r * cell_h
            y2 = pad + (r+1) * cell_h

            if random.random() < p_landmark:
                landmarks.append((l_id, int(x), int(y1)))
                l_id += 1
            if random.random() < p_landmark:
                landmarks.append((l_id, int(x), int(y2)))
                l_id += 1

    return landmarks

    
def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def draw_random_obstacles(
    screen,
    wall_list,
    horiz,
    vert,
    pad=20,
    grid_w=16,
    grid_h=9,
    wall_thickness=1,
    n_obstacles=100,
    obstacle_mu=7.5,
    obstacle_sigma=1.5,
    obstacle_color=(0,0,0)
):
    sw, sh = screen.get_size()
    map_w = sw - 2*pad
    map_h = sh - 2*pad
    cell_w = map_w / grid_w
    cell_h = map_h / grid_h

    # collect all cell‐interiors that have no walls on any of their 4 borders
    free_areas = []
    for r in range(grid_h):
        for c in range(grid_w):
            if (horiz[r][c] or horiz[r+1][c] or
                vert[r][c] or vert[r][c+1]):
                continue
            interior = pygame.Rect(
                int(pad + c*cell_w + wall_thickness),
                int(pad + r*cell_h + wall_thickness),
                int(cell_w - 2*wall_thickness),
                int(cell_h - 2*wall_thickness)
            )
            free_areas.append(interior)

    # scatter obstacles in those areas
    obstacle_list = []
    attempts = 0
    while len(obstacle_list) < n_obstacles and attempts < n_obstacles*10:
        attempts += 1
        area = random.choice(free_areas)
        #w = clamp(random.gauss(obstacle_mu, obstacle_sigma), 5, 10)
        #h = clamp(random.gauss(obstacle_mu, obstacle_sigma), 5, 10)
        w = wall_thickness * 2
        h = wall_thickness * 2
        ox = random.uniform(area.x, area.x + area.width  - w)
        oy = random.uniform(area.y, area.y + area.height - h)
        orect = pygame.Rect(int(ox), int(oy), int(w), int(h))

        # reject if it bumps any wall-segment
        if any(orect.colliderect(wr) for wr in wall_list):
            continue

        pygame.draw.rect(screen, obstacle_color, orect)
        obstacle_list.append(orect)

    return obstacle_list

def draw_cells(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect( cell[0] * block_width, cell[1] * block_width, block_width, block_width)
        pygame.draw.rect(screen, color, rect, 1)  # Draw the cell with a red border

def draw_cells_filled(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect( cell[0] * block_width, cell[1] * block_width, block_width, block_width)
        pygame.draw.rect(screen, color, rect)  # Draw the cell with a red border