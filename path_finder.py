import math
import random
import heapq
import pygame

def find_path(
    grid_probability,
    start_xy, goal_xy,
    grid_size,
    robot_radius=0,
    safety_param=1.2,
    occ_thresh=0.6,
    draw=False,
    surface=None,
    line_color=(255,0,0),
    line_width=5,
):

    n_cols, n_rows = grid_probability.shape

    # convert start/goal to grid coords
    sx, sy = int(start_xy[0] // grid_size), int(start_xy[1] // grid_size)
    gx, gy = int(goal_xy[0]  // grid_size), int(goal_xy[1]  // grid_size)

    # find all raw occupied cells
    raw_occ = {
        (i,j)
        for i in range(n_cols)
        for j in range(n_rows)
        if grid_probability[i,j] > occ_thresh
    }

    # extend them by robot_radius
    r_cells = math.ceil((robot_radius * safety_param) / grid_size)
    occupied = set(raw_occ)
    for (i,j) in raw_occ:
        for di in range(-r_cells, r_cells+1):
            for dj in range(-r_cells, r_cells+1):
                ii, jj = i+di, j+dj
                if 0 <= ii < n_cols and 0 <= jj < n_rows:
                    # only if within distance
                    if math.hypot(di, dj) * grid_size <= robot_radius * safety_param:
                        occupied.add((ii, jj))

    # A* search
    open_set = []
    start_h = math.hypot(gx - sx, gy - sy)
    heapq.heappush(open_set, (start_h, 0, (sx,sy), None))
    came_from = {}
    cost_so_far = {(sx,sy): 0}

    path = []
    while open_set:
        f, g, (i,j), parent = heapq.heappop(open_set)

        # check if we reached goal
        if (i,j) == (gx,gy):
            came_from[(i,j)] = parent
            path = create_path(came_from, (sx,sy), (gx,gy), grid_size)
            break

        if parent is not None:
            came_from[(i,j)] = parent

        # shuffle neighbors
        nbs = list(neighbors(i, j, n_cols, n_rows))
        random.shuffle(nbs)
        for ni,nj in nbs:
            if (ni,nj) in occupied:
                continue
            step = math.hypot(ni - i, nj - j)
            new_g = g + step
            if (ni,nj) not in cost_so_far or new_g < cost_so_far[(ni,nj)]:
                cost_so_far[(ni,nj)] = new_g
                h = math.hypot(gx - ni, gy - nj)
                heapq.heappush(open_set, (new_g + h, new_g, (ni,nj), (i,j)))

    if not path:
        return []

    # draw
    if draw and (surface is not None) and (len(path) > 1):
        pygame.draw.lines(surface, line_color, False, path, line_width)

    return path


def neighbors(i, j, n_cols, n_rows):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0: 
                continue
            ni, nj = i+di, j+dj
            if 0 <= ni < n_cols and 0 <= nj < n_rows:
                yield (ni, nj)


def create_path(came_from, start, goal, grid_size):
    node = goal
    rev = []
    while node != start:
        rev.append(node)
        node = came_from.get(node)
        if node is None:
            break
    rev.append(start)
    rev.reverse()
    return [((i+0.5)*grid_size, (j+0.5)*grid_size) for i,j in rev]
