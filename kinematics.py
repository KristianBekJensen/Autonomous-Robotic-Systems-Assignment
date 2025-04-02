import numpy as np
from scipy.integrate import odeint

# Robot parameters
#wheel_radius = 0.05
axel_lenght = 3

def differential_drive_kinematics(state, t, v_left, v_right):
    x, y, direction = state
    v = (v_left + v_right)/2 
    rotation = (v_right - v_left)/axel_lenght 
    dxdt = v*np.cos(direction)
    dydt = v*np.sin(direction)

    return [dxdt, dydt, rotation]


def update_robot_state(initial_state, t, v_left, v_right):
    return odeint(differential_drive_kinematics, initial_state, t, args=(v_left, v_right))

