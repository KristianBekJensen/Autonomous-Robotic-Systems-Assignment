import math
import numpy as np

def line_through_grid(start, end, grid_size, number_x_grids, number_y_grids):
    
    #Determine all grid cells that a line from start to end passes through.
    
    # Convert world coordinates to grid indices 
    def world_to_grid(point):
        x, y = point

        # Get grid indices
        i = int(x / grid_size)
        j = int(y / grid_size)
        
        # Ensure indices are within grid bounds
        i = max(0, min(i, number_x_grids - 1))
        j = max(0, min(j, number_y_grids - 1))
        
        return i, j
    
    # Start and end coordinates
    x0, y0 = start
    x1, y1 = end
    
    # Get starting and ending grid cells
    start_cell = world_to_grid(start)
    end_cell = world_to_grid(end)
    
    # Initialize result list with starting cell
    cells = [start_cell]
    
    # Line direction and length
    dx = x1 - x0
    dy = y1 - y0
    
    # Line length (Manhattan distance for grid)
    steps = max(abs(dx), abs(dy))
    
    if steps == 0:  # Start and end are in the same cell
        return cells[:-1], cells[-1] if cells else ([], start_cell)
    
    # Step increments
    x_inc = dx / steps
    y_inc = dy / steps
    
    # Current position (starting just after the initial position)
    x = x0 + 0.5 * x_inc
    y = y0 + 0.5 * y_inc
    
    # Traverse the line
    for _ in range(int(steps)):
        x += x_inc
        y += y_inc
        
        # Get grid cell for current position
        cell = world_to_grid((x, y))
        
        # Add cell if it's not already in the list
        if cell != cells[-1]:
            cells.append(cell)
    
    # Return all cells except the last one, and the last cell separately
    return cells[0:-1], cells[-1] if cells else ([], start_cell)

def get_observed_cells(robot, grid_size, number_x_grids, number_y_grids):
    """
    Determines which grid cells are observed as free or occupied based on sensor data.
    
    Parameters:
    - robot: The robot object containing position and sensor information
    - grid_size: Size of each grid cell
    - grid_width: Number of grid cells in x-direction
    - grid_height: Number of grid cells in y-direction
    
    Returns:
    - Two sets: free cells and occupied cells
    """
    estimated_x, estimated_y, estimated_theta = robot.estimated_poses[-1]
    free_cells = set()
    occupied_cells = set()
    for i in range(robot.num_sensors):
        sensor_theta = (estimated_theta + (2*np.pi/robot.num_sensors*i)) % (2*np.pi)
        
        end_point = (
            estimated_x + (robot.sensor_values[i] + robot.radius) * math.cos(sensor_theta),
            estimated_y + (robot.sensor_values[i] + robot.radius) * math.sin(sensor_theta)
        )
        
        free_cell, last_cell = line_through_grid(
            (estimated_x, estimated_y),
            end_point,
            grid_size,
            number_x_grids,
            number_y_grids
        )
        
        # Add free cells
        for cell in free_cell:
            free_cells.add(cell)
            
        # Add last cell as either free or occupied
        if robot.sensor_values[i] == robot.max_sensor_range:
            free_cells.add(last_cell)
        else:
            occupied_cells.add(last_cell)
    
    return free_cells, occupied_cells

def log_odds_to_prob(log_odds):
    return 1 - (1 / (1 + np.exp(log_odds)))

def probs_to_grey_scale(prob):
    return (prob * 255).astype(np.uint8)

def calculate_mapping_accuracy(estimated_grid, walls, obstacles, screen_width, screen_height, grid_size):
    """
    Placeholder function to calculate mapping accuracy.
    Replace this with the actual implementation.
    """
    pass
    grid_real = np.zeros((int(screen_width  / grid_size),int(screen_height  / grid_size)))
    for wall in walls:
        (left, top, width, height) = wall
        for i in range(int(left/grid_size), int((left+width)/grid_size)):
            for j in range(int(top/grid_size), int((top+height)/grid_size)):
                grid_real[i][j] = 1
    for obstacle in obstacles:
        (left, top, width, height) = obstacle
        for i in range(int(left/grid_size), int((left+width)/grid_size)):
            for j in range(int(top/grid_size), int((top+height)/grid_size)):
                grid_real[i][j] = 1
    
    average_error = 0

    for i in range(len(estimated_grid)):
        for j in range(len(estimated_grid[i])):
            error = abs(grid_real[i][j] - estimated_grid[i][j])
            average_error += error
    average_error /= (len(estimated_grid) * len(estimated_grid[0]))
    return average_error




    


