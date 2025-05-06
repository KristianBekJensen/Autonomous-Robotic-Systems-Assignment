import numpy as np
from math import cos, sin, atan2, sqrt

def get_landmark_measurements(landmarks, robot_pose):
    """
    Get measurements to landmarks (range, bearing, signature)
    
    Parameters:
    -----------
    robot_pose : numpy.ndarray
        Robot pose [x, y, theta]
    add_noise : bool
        Whether to add noise to measurements
    noise_std : float
        Standard deviation of measurement noise
        
    Returns:
    --------
    measurements : list of tuples
        List of measurements [(range1, bearing1, id1), (range2, bearing2, id2), ...]
    """
    x, y, theta = robot_pose
    measurements = []
    
    for (i, m_x, m_y) in landmarks:
        # Calculate range
        r = sqrt((m_x - x)**2 + (m_y - y)**2)
        
        # Calculate bearing
        phi = atan2(m_y - y, m_x - x) - theta
        # Normalize angle to [0, 2pi]
        phi = (phi + np.pi) % (2 * np.pi)
        
        # Use landmark index as signature
        s = i
        
        measurements.append((r, phi, s))
    
    return measurements


def phi(point, landmark, theta):
    x, y = point
    m_x, m_y = landmark
        
    # Calculate bearing
    phi = atan2(m_y - y, m_x - x) - theta
    
    # Normalize angle to [0, 2pi]
    return (phi + np.pi) % (2 * np.pi) 

def two_point_triangulate(measurements, landmarks, robot, position_noise, theta_noise):
    _, _, theta = robot

    if len(measurements) > 1:
        a_r, a_phi, a_s =  measurements[0]
        b_r, b_phi, b_s =  measurements[1]

        lookup = {idx: (x, y) for idx, x, y in landmarks}
        a_x, a_y = lookup[a_s]
        b_x, b_y = lookup[b_s]

        import math

        def intersect_two_circles(x1,y1,r1, x2,y2,r2):
            dx, dy = x2-x1, y2-y1
            d = math.hypot(dx,dy)
            if d > r1+r2 or d < abs(r1-r2) or d == 0:
                return [(0,0), (0,0)]  # no solutions or infinite
            a = (r1*r1 - r2*r2 + d*d) / (2*d)
            h_sq = r1*r1 - a*a
            if h_sq < 0:
                h_sq = 0  # correct small negative due to floating point errors
            h = math.sqrt(h_sq)

            xm = x1 + a*dx/d
            ym = y1 + a*dy/d
            rx = -dy * (h/d)
            ry =  dx * (h/d)
            return [(xm+rx, ym+ry), (xm-rx, ym-ry)]

        p1, p2 =  intersect_two_circles(a_x, a_y, a_r, b_x, b_y, b_r)

        p1_Aphi = phi(p1, (a_x, a_y), theta)
        p2_Aphi = phi(p2, (a_x, a_y), theta)
        p1_Bphi = phi(p1, (b_x, b_y), theta)
        p2_Bphi = phi(p2, (b_x, b_y), theta)

        if abs(p1_Aphi - a_phi) < abs(p2_Aphi - a_phi) and abs(p1_Bphi - b_phi) < abs(p2_Bphi - b_phi):
            chosen, other = p1, p2
        elif abs(p1_Aphi - a_phi) > abs(p2_Aphi - a_phi) and abs(p1_Bphi - b_phi) > abs(p2_Bphi - b_phi):
            chosen, other = p2, p1
        try:

            noisy_chosen = (
                chosen[0] + np.random.normal(0, position_noise),
                chosen[1] + np.random.normal(0, position_noise)
            )
            noisy_other = (
                other[0] + np.random.normal(0, position_noise),
                other[1] + np.random.normal(0, position_noise)
            )

            noisy_theta = theta + np.random.normal(0, theta_noise)
            return noisy_chosen, noisy_other, noisy_theta
        except:
            return (0,0), (0,0), 0
    
    return (0,0), (0,0), 0