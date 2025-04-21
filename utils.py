import math
import pygame

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
