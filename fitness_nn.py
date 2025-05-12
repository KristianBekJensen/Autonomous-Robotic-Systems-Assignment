import math

def fitness(num_collisions: int,
            num_time_steps: int,
            dist_to_target: float,
            collision_weight: float = 1.0,
            time_weight: float = 1.0,
            dist_weight: float = 20.0) -> float:

    return (collision_weight * num_collisions
            + time_weight * num_time_steps
            + dist_weight * dist_to_target)


def distance_to_target(position: tuple[float, float], target: tuple[float, float]) -> float:
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    return math.hypot(dx, dy)
