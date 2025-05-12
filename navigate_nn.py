import numpy as np
from maze_solver import NeuralController

def navigate(keys, v_left, v_right):
    if keys.get('w', False):
        if v_left != v_right:
            v_right = v_left = max(v_left, v_right)
        else:
            v_right += 0.2;  v_left += 0.2
    elif keys.get('s', False):
        v_left = max(0, v_left - 0.2)
        v_right = max(0, v_right - 0.2)
    elif keys.get('d', False):
        v_right *= 0.98
    elif keys.get('a', False):
        v_left  *= 0.98
    elif keys.get('e', False):
        if v_left == 0 and v_right == 0:
            v_left, v_right =  0.3, -0.3
    elif keys.get('q', False):
        if v_left == 0 and v_right == 0:
            v_left, v_right = -0.3,  0.3
    else:
        # keep them in sync
        if v_left < 0 or v_right < 0:
            v_left = v_right = 0
        elif v_right >= v_left:
            v_left = v_right
        else:
            v_right = v_left

    return v_left, v_right


def phenome_navigate(phenome,
                     sensor_vals,
                     v_left,
                     v_right,
                     min_speed,
                     max_speed,
                     angle_to_path,
                     wheel_inputs=None,   # now unused
                     angle_inputs=None):  # now unused
    """
    Float‐vector controller: build MLP, normalize inputs,
    run forward, then rescale outputs to [min_speed,max_speed].
    """

    # 1) Build the controller from the genotype
    #    (input_size == len(sensor_vals) + 2 wheel speeds + 1 heading error)
    input_size  = len(sensor_vals) + 2 + 1
    hidden_size = 5
    output_size = 2

    controller = NeuralController(
        genotype=phenome,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )

    # 2) Assemble and normalize the input vector
    x = np.zeros(input_size, dtype=float)

    # a) sensors in [0,1]
    max_sensor = 100.0
    x[:len(sensor_vals)] = np.array(sensor_vals) / max_sensor

    # b) wheel speeds in [0,1]
    x[len(sensor_vals) + 0] = (v_left  - min_speed) / (max_speed - min_speed)
    x[len(sensor_vals) + 1] = (v_right - min_speed) / (max_speed - min_speed)

    # c) heading‐error in [0,1]
    x[len(sensor_vals) + 2] = angle_to_path / (2 * np.pi)

    # 3) Forward pass → 2 outputs in [–1,1]
    out = controller.forward(x)

    # 4) Rescale back to [min_speed,max_speed]
    new_v_left  = min_speed + (out[0] + 1) / 2 * (max_speed - min_speed)
    new_v_right = min_speed + (out[1] + 1) / 2 * (max_speed - min_speed)

    return new_v_left, new_v_right
