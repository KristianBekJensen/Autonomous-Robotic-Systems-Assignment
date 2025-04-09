import numpy as np

# Robot parameters
#wheel_radius = 0.05
axel_length = 20

def differential_drive_kinematics(state, t, v_left, v_right):
    x, y, direction = state
    v = (v_left + v_right)/2 
    rotation = (v_right - v_left)/axel_length 
    dxdt = v*np.cos(direction)
    dydt = v*np.sin(direction)

    return [dxdt, dydt, -rotation]