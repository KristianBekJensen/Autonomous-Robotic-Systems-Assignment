import math
import pygame
import numpy as np

def draw_dashed_lines(surface, color, points, width=2,dash_length=8, space_length=8):
    """
    Draw a dashed polyline through `points`.
      • dash_length: length in pixels of each dash
      • space_length: length in pixels of each gap
    """
    # 1) Build a list of segments with their unit‐directions and cumulative lengths
    segs = []           # each: (x1, y1, vx, vy, seg_len)
    cum_lengths = [0]   # cumulative length at the start of each segment
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue
        vx, vy = dx/seg_len, dy/seg_len
        segs.append((x1, y1, vx, vy, seg_len))
        cum_lengths.append(cum_lengths[-1] + seg_len)

    total_length = cum_lengths[-1]
    if total_length == 0:
        return

    # 2) Helper to get a point at a given distance along the entire polyline
    def point_at(dist):
        if dist <= 0:
            return points[0]
        if dist >= total_length:
            return points[-1]
        # find which segment contains this dist
        for i, seg_start in enumerate(cum_lengths[:-1]):
            seg_end = cum_lengths[i+1]
            if seg_start <= dist < seg_end:
                x1, y1, vx, vy, seg_len = segs[i]
                local = dist - seg_start
                return (x1 + vx*local, y1 + vy*local)

    # 3) Walk along the polyline, alternating dash/gap
    pattern = dash_length + space_length
    dist = 0.0
    draw = True
    while dist < total_length:
        this_len = dash_length if draw else space_length
        start_pt = point_at(dist)
        end_pt   = point_at(min(dist + this_len, total_length))
        if draw:
            pygame.draw.line(surface, color, start_pt, end_pt, width)
        dist += this_len
        draw = not draw

def draw_covariance_ellipse(surface, mean, cov, n_std=2.0,
                            num_points=36, color=(255,255,0), width=1):
    """
    Draw an n_std‑sigma covariance ellipse in pygame.

    surface:    your pygame.Surface
    mean:       (x, y) tuple in pixel coords
    cov:        2×2 numpy covariance matrix
    n_std:      how many standard deviations (e.g. 2 for ~95% region)
    num_points: how many segments to approximate the ellipse
    color:      RGB tuple
    width:      line thickness
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # radii of the ellipse axes
    rx = n_std * math.sqrt(vals[0])
    ry = n_std * math.sqrt(vals[1])
    # rotation of the ellipse (angle of the largest eigenvector)
    angle = math.atan2(vecs[1,0], vecs[0,0])
    ca, sa = math.cos(angle), math.sin(angle)
    cx, cy = mean

    pts = []
    for i in range(num_points):
        θ = 2 * math.pi * i / num_points
        x_ = rx * math.cos(θ)
        y_ = ry * math.sin(θ)
        # rotate then translate
        xr =  x_*ca - y_*sa + cx
        yr =  x_*sa + y_*ca + cy
        pts.append((xr, yr))

    # draw as a polygon
    pygame.draw.polygon(surface, color, pts, width)