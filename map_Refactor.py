import pygame

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