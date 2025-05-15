import math
import pygame
import numpy as np

def draw_covariance_ellipse(surface, mean, cov, n_std=2.0, num_points=36, color=(255,255,0), width=1):
    """ A helper function to visualize uncertainty regions in robot movement """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # radii of the ellipse axes
    # but here since vals[0] and vals[1] are equal it becomes a circle
    rx = n_std * math.sqrt(vals[0])
    ry = n_std * math.sqrt(vals[1])
    # rotation of the ellipse (angle of the largest eigenvector)
    angle = math.atan2(vecs[1,0], vecs[0,0])
    ca, sa = math.cos(angle), math.sin(angle)
    cx, cy = mean

    pts = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x_ = rx * math.cos(theta)
        y_ = ry * math.sin(theta)
        # rotate then translate
        xr = x_*ca - y_*sa + cx
        yr = x_*sa + y_*ca + cy
        pts.append((xr, yr))

    # draw as a polygon
    pygame.draw.polygon(surface, color, pts, width)