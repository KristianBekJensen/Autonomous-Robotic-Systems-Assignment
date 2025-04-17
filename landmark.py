import numpy as np
from math import cos, sin, atan2, sqrt

def get_landmark_measurements(landmarks, robot_pose, add_noise=True, noise_std=0.1):
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
        
        # Normalize angle to [-pi, pi]
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        
        # Use landmark index as signature
        s = i
        
        # Add noise if requested
        if add_noise:
            r += np.random.normal(0, noise_std)
            phi += np.random.normal(0, noise_std)
        
        measurements.append((r, phi, s))
    
    return measurements

def triangulate_position(measurements, landmarks):
    """
    Calculate robot position from landmark measurements using triangulation
    This is a simplified calculation that averages positions from all landmarks
    
    Parameters:
    -----------
    measurements : list of tuples
        List of measurements [(range1, bearing1, id1), (range2, bearing2, id2), ...]
    landmarks : list of tuples
        List of landmark positions
        
    Returns:
    --------
    position : numpy.ndarray
        Estimated robot position [x, y, theta]
    """
    x_sum, y_sum, theta_sum = 0, 0, 0
    count = 0
    
    for r, phi, s in measurements:

        lookup = {idx: (x, y) for idx, x, y in landmarks}
        m_x, m_y = lookup[s]
        
        # Estimate position based on range and bearing
        count += 1
        
        # For theta estimation, we use the bearing measurement
        # and the known positions of landmarks
        estimated_theta = atan2(m_y - y_sum/max(1, count-1), 
                                m_x - x_sum/max(1, count-1)) - phi
        
        # Normalize theta
        estimated_theta = (estimated_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Update sums
        theta_sum += estimated_theta
        
        # Estimate position based on range and bearing
        x_sum += m_x - r * cos(phi + estimated_theta)
        y_sum += m_y - r * sin(phi + estimated_theta)
    
    if count > 0:
        return np.array([x_sum/count, y_sum/count, theta_sum/count]) # this is z
    else:
        return np.array([0, 0, 0])