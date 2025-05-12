import math

def fitness(num_collisions: int,
            num_time_steps: int,
            dist_to_target: float,
            collision_weight: float = 100.0,
            time_weight: float = 1.0,
            dist_weight: float = 1.0) -> float:
    """
    Scalar fitness: higher costs are worse (we're minimizing).
    
    Args:
        num_collisions:   total collision count during the trial
        num_time_steps:   number of simulation steps taken
        dist_to_target:   Euclidean distance from robot to goal at end
        collision_weight: penalty per collision
        time_weight:      penalty per time step
        dist_weight:      penalty per unit distance remaining

    Returns:
        A single scalar score (lower is better).
    """
    return (collision_weight * num_collisions
            + time_weight      * num_time_steps
            + dist_weight      * dist_to_target)


def distance_to_target(position: tuple[float, float],
                       target:   tuple[float, float]) -> float:
    """
    Euclidean distance between two 2D points.
    """
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    return math.hypot(dx, dy)
