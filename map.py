import pygame
import random
import numpy as np

SEED = 44
random.seed(SEED)
np.random.seed(SEED)

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def draw_map(screen, num_blocks_w, num_blocks_h, 
    pad=20, 
    wall_color=(0, 0, 0), 
    wall_h_prob=0.2,
    wall_v_prob=0.2,
    wall_thickness=1,
    p_landmark=0.25,
    n_obstacles=100,
    obstacle_mu=7.5,
    obstacle_sigma=1.5,
    obstacle_color=(0,0,0)
):
    screen_w, screen_h = screen.get_size()
    map_w = screen_w - 2 * pad
    map_h = screen_h - 2 * pad
    cell_w = map_w / num_blocks_w
    cell_h = map_h / num_blocks_h

    horiz = [
        [1 if random.random() < wall_h_prob else 0
        for _ in range(num_blocks_w)]
        for _ in range(num_blocks_h+1)
    ]
    vert = [
        [1 if random.random() < wall_v_prob else 0
        for _ in range(num_blocks_w+1)]
        for _ in range(num_blocks_h)
    ]

    # force all outer borders ON
    # top and bottom horizontal walls:
    for c in range(num_blocks_w):
        horiz[0][c] = 1 # top edge of map
        horiz[num_blocks_h][c] = 1 # bottom edge of map

    # left and right vertical walls:
    for r in range(num_blocks_h):
        vert[r][0] = 1 # left edge of map
        vert[r][num_blocks_w] = 1 # right edge of map

    ## WALLS
    wall_list = []

    # Horizontal walls
    for r in range(num_blocks_h + 1):
        y = pad + r * cell_h
        for c, present in enumerate(horiz[r]):
            if present:
                x = pad + c * cell_w
                rect = pygame.Rect(int(x), int(y), int(cell_w), wall_thickness)
                pygame.draw.rect(screen, wall_color, rect)
                wall_list.append(rect)

    # Vertical walls
    for r in range(num_blocks_h):
        y = pad + r * cell_h
        for c, present in enumerate(vert[r]):
            if present:
                x = pad + c * cell_w
                rect = pygame.Rect(int(x), int(y), wall_thickness, int(cell_h))
                pygame.draw.rect(screen, wall_color, rect)
                wall_list.append(rect)


    ## LANDMARKS
    landmarks = []
    l_id = 0

    # Horizontal segments: r in [0..grid_h], c in [0..grid_w-1]
    for r in range(num_blocks_h + 1):
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
    for r in range(num_blocks_h):
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


    # collect all cell‐interiors that have no walls on any of their 4 borders
    free_areas = []
    for r in range(num_blocks_h):
        for c in range(num_blocks_w):
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

    ## OBSTACLES scattered in the map
    obstacle_list = []
    attempts = 0
    while len(obstacle_list) < n_obstacles and attempts < n_obstacles*10:
        attempts += 1
        area = random.choice(free_areas)
        w = clamp(random.gauss(obstacle_mu, obstacle_sigma), 5, 10)
        h = clamp(random.gauss(obstacle_mu, obstacle_sigma), 5, 10)
        ox = random.uniform(area.x, area.x + area.width  - w)
        oy = random.uniform(area.y, area.y + area.height - h)
        orect = pygame.Rect(int(ox), int(oy), int(w), int(h))

        # reject if it bumps any wall-segment
        if any(orect.colliderect(wr) for wr in wall_list):
            continue

        pygame.draw.rect(screen, obstacle_color, orect)
        obstacle_list.append(orect)

    return wall_list, landmarks, obstacle_list


def draw_cells(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect( cell[0] * block_width, cell[1] * block_width, block_width, block_width)
        pygame.draw.rect(screen, color, rect, 1)  # Draw the cell with a red border

def draw_cells_filled(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect( cell[0] * block_width, cell[1] * block_width, block_width, block_width)
        pygame.draw.rect(screen, color, rect)  # Draw the cell with a red border