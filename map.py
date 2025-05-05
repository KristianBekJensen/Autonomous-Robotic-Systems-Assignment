import pygame
import random

def draw_map(
    screen,
    horiz,
    vert,
    pad=20,
    grid_w=16,
    grid_h=9,
    wall_color=(0, 0, 0),
    wall_thickness=1
):
    screen_w, screen_h = screen.get_size()
    map_w = screen_w  - 2 * pad
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
                rect = pygame.Rect(
                    int(x), int(y),
                    int(cell_w), wall_thickness
                )
                pygame.draw.rect(screen, wall_color, rect)
                wall_list.append(rect)

    # Vertical walls
    for r in range(grid_h):
        y = pad + r * cell_h
        for c, present in enumerate(vert[r]):
            if present:
                x = pad + c * cell_w
                rect = pygame.Rect(
                    int(x), int(y),
                    wall_thickness, int(cell_h)
                )
                pygame.draw.rect(screen, wall_color, rect)
                wall_list.append(rect)

    return wall_list


def compute_landmarks(
    horiz,
    vert,
    screen,
    pad=20,
    grid_w=16,
    grid_h=9
):
    screen_w, screen_h = screen.get_size()
    map_w = screen_w  - 2 * pad
    map_h = screen_h - 2 * pad
    cell_w = map_w / grid_w
    cell_h = map_h / grid_h

    landmarks = []
    lid = 0

    for r in range(grid_h + 1):
        for c in range(grid_w + 1):
            # only access horiz[r-1][c] when c < grid_w
            above = (r > 0       and c < grid_w and horiz[r-1][c] == 1)
            # only access horiz[r][c]   when c < grid_w
            below = (r < grid_h and c < grid_w and horiz[r][c]   == 1)
            # only access vert[r][c-1]  when r < grid_h
            left  = (c > 0       and r < grid_h and vert[r][c-1] == 1)
            # only access vert[r][c]    when r < grid_h and c < len(vert[r])
            right = (r < grid_h and c < len(vert[r]) and vert[r][c]   == 1)

            # if any orthogonal pair is present, mark a corner
            if (above and left) or (above and right) or (below and left) or (below and right):
                px = pad + c * cell_w
                py = pad + r * cell_h
                landmarks.append((lid, int(px), int(py)))
                lid += 1

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
    """
    - screen: your pygame surface
    - wall_list: list of pygame.Rect returned by draw_map()
    - horiz:      (grid_h+1)×grid_w matrix of horizontal walls
    - vert:       grid_h×(grid_w+1)  matrix of vertical   walls
    - pad, grid_w, grid_h, wall_thickness: same as draw_map()
    - n_obstacles, obstacle_mu/sigma: Gaussian obstacle sizes
    """
    sw, sh = screen.get_size()
    map_w = sw  - 2*pad
    map_h = sh  - 2*pad
    cell_w = map_w / grid_w
    cell_h = map_h / grid_h

    # 1) collect all cell‐interiors that have NO walls on any of their 4 borders
    free_areas = []
    for r in range(grid_h):
        for c in range(grid_w):
            if (horiz[r][c]   or horiz[r+1][c] or
                vert[r][c]    or vert[r][c+1]):
                continue
            interior = pygame.Rect(
                int(pad + c*cell_w + wall_thickness),
                int(pad + r*cell_h + wall_thickness),
                int(cell_w  - 2*wall_thickness),
                int(cell_h  - 2*wall_thickness)
            )
            free_areas.append(interior)

    # 2) scatter obstacles in those areas
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

    return obstacle_list

def draw_cells(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect(
            cell[0] * block_width,
            cell[1] * block_width,
            block_width,
            block_width
        )
        pygame.draw.rect(screen, color, rect, 1)  # Draw the cell with a red border

def draw_cells_filled(cells, screen, block_width, color = "red"):
    for cell in cells:
        rect = pygame.Rect(
            cell[0] * block_width,
            cell[1] * block_width,
            block_width,
            block_width
        )
        pygame.draw.rect(screen, color, rect)  # Draw the cell with a red border