def line_through_grid(start, end, grid_size):
    """
    Determines all grid cells that a line from start to end passes through.
    
    Parameters:
    - start: (x, y) starting point coordinates
    - end: (x, y) ending point coordinates
    - grid_size: Size of each grid cell
    
    Returns:
    - List of (i, j) grid cell indices the line passes through
    """
    # Convert world coordinates to grid indices
    def world_to_grid(point):
        x, y = point
        # Integer division to get grid indices
        i = int(y / grid_size)
        j = int(x / grid_size)
        return i, j
    
    # Extract coordinates
    x0, y0 = start
    x1, y1 = end
    
    # Get starting and ending grid cells
    start_cell = world_to_grid(start)
    end_cell = world_to_grid(end)
    
    # Initialize result list with starting cell
    cells = [start_cell]
    
    # Calculate line direction and length
    dx = x1 - x0
    dy = y1 - y0
    
    # Line length (using Manhattan distance for grid traversal)
    steps = max(abs(dx), abs(dy))
    
    if steps == 0:  # Start and end are in the same cell
        return cells
    
    # Calculate step increments
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
    
    return cells[0:-1], cells[-1]
