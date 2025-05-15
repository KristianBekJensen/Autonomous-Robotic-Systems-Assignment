import pygame
import random
import numpy as np

def draw_map(
    screen, 
    num_blocks_w, 
    num_blocks_h, 
    pad=20,
    wall_color=(0, 0, 0), 
    wall_h_prob=0.2,
    wall_v_prob=0.2,
    wall_thickness=1,
    p_landmark=0.25,
    n_obstacles=100,
    obstacle_color=(0,0,0),
    random_seed=44,
):
    """ generate and draw a map including walls, obstacles, and landmarks """

    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # calculate map width and height based on its padding from screen sides
    screen_w, screen_h = screen.get_size()
    map_w = screen_w - 2 * pad
    map_h = screen_h - 2 * pad

    # calculate map cell sizes 
    cell_w = map_w / num_blocks_w
    cell_h = map_h / num_blocks_h

    # from a matrix of size num_blocks_w x num_blocks_h+1 which represents
    # all horizontal walls, randomly turn some elements to 1 and draw that wall
    horiz = [
        [1 if random.random() < wall_h_prob else 0 for _ in range(num_blocks_w)]
        for _ in range(num_blocks_h+1)
    ]

    # from a matrix of size num_blocks_w+1 x num_blocks_h which represents
    # all vertical walls, randomly turn some elements to 1 and draw that wall
    vert = [
        [1 if random.random() < wall_v_prob else 0 for _ in range(num_blocks_w+1)]
        for _ in range(num_blocks_h)
    ]

    # force all outer walls to be present
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

    # draw (present)horizontal walls
    for r in range(num_blocks_h + 1):
        y = pad + r * cell_h
        for c, present in enumerate(horiz[r]):
            if present:
                x = pad + c * cell_w
                rect = pygame.Rect(int(x), int(y), int(cell_w), wall_thickness)
                pygame.draw.rect(screen, wall_color, rect)
                wall_list.append(rect)

    # draw (present)vertical walls
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

    # each horizontal wall endpoint can potentially have a landmark
    for r in range(num_blocks_h + 1):
        y = pad + r * cell_h
        for c, present in enumerate(horiz[r]):
            if not present:
                continue

            # calculate endpoints
            x1, y1 = pad + c * cell_w, y
            x2, y2 = pad + (c+1) * cell_w, y

            # each endpoint has a chance of p_landmark to have a landmark
            if random.random() < p_landmark:
                landmarks.append((l_id, int(x1), int(y1)))
                l_id += 1
            if random.random() < p_landmark:
                landmarks.append((l_id, int(x2), int(y2)))
                l_id += 1

    # each vertical wall endpoint can potentially have a landmark
    for r in range(num_blocks_h):
        x = pad + c * cell_w
        for c, present in enumerate(vert[r]):
            if not present:
                continue

            # calculate endpoints
            x = pad + c * cell_w
            y1 = pad + r * cell_h
            y2 = pad + (r+1) * cell_h

            # each endpoint has a chance of p_landmark to have a landmark
            if random.random() < p_landmark:
                landmarks.append((l_id, int(x), int(y1)))
                l_id += 1
            if random.random() < p_landmark:
                landmarks.append((l_id, int(x), int(y2)))
                l_id += 1


    ## OBSTACLES scattered in the map
    # first find free areas in the map that we can place obstacles in
    free_areas = []
    for r in range(num_blocks_h):
        for c in range(num_blocks_w):
            interior = pygame.Rect(
                int(pad + c*cell_w + wall_thickness),
                int(pad + r*cell_h + wall_thickness),
                int(cell_w - 2*wall_thickness),
                int(cell_h - 2*wall_thickness)
            )
            free_areas.append(interior)

    obstacle_list = []
    attempts = 0
    # try to put obstacles randomly but avoid walls
    while len(obstacle_list) < n_obstacles and attempts < n_obstacles*10:
        attempts += 1
        area = random.choice(free_areas)
        w = 2 * wall_thickness
        h = 2 * wall_thickness
        ox = random.uniform(area.x, area.x + area.width  - w)
        oy = random.uniform(area.y, area.y + area.height - h)
        orect = pygame.Rect(int(ox), int(oy), int(w), int(h))

        # reject if it collides any wall
        if any(orect.colliderect(wr) for wr in wall_list):
            continue

        pygame.draw.rect(screen, obstacle_color, orect)
        obstacle_list.append(orect)

    return wall_list, landmarks, obstacle_list


def draw_cells(cells, screen, block_width, color = "red"):
    """ function to draw grid cells in sensor range when its visualization param is true """
    for cell in cells:
        rect = pygame.Rect( cell[0] * block_width, cell[1] * block_width, block_width, block_width)
        pygame.draw.rect(screen, color, rect, 1)  # Draw the cell with a red border