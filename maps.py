import pygame

def draw_map(screen, edges, wall_coords, wall_thickness):
    border_thickness = 2
    border_color = (0, 0, 0)
    
    # Edges of the map in this order: top-left, top-right, bottom-right, bottom-left.
    top_left = edges[0]
    bottom_right = edges[2]
    
    # Calculate the inner map dimensions.
    inner_width = bottom_right[0] - top_left[0]
    inner_height = bottom_right[1] - top_left[1]
    
    # Create and draw border rectangles.
    top_rect = pygame.Rect(top_left[0], top_left[1], inner_width, border_thickness)
    bottom_rect = pygame.Rect(top_left[0], bottom_right[1] - border_thickness, inner_width, border_thickness)
    left_rect = pygame.Rect(top_left[0], top_left[1], border_thickness, inner_height)
    right_rect = pygame.Rect(bottom_right[0] - border_thickness, top_left[1], border_thickness, inner_height)
    
    pygame.draw.rect(screen, border_color, top_rect)
    pygame.draw.rect(screen, border_color, bottom_rect)
    pygame.draw.rect(screen, border_color, left_rect)
    pygame.draw.rect(screen, border_color, right_rect)
    
    wall_color = (0, 0, 0)
    wall_rects = [top_rect, bottom_rect, left_rect, right_rect]
    
    # Process each wall from the dictionary.
    for wall_id, wall in wall_coords.items():
        (x1, y1), (x2, y2) = wall
        
        # Create a pygame.Rect based on the orientation of the wall.
        if y1 == y2:
            # Horizontal wall.
            x = min(x1, x2)
            width = abs(x2 - x1)
            rect = pygame.Rect(x, y1, width, wall_thickness)
        elif x1 == x2:
            # Vertical wall.
            y = min(y1, y2)
            height = abs(y2 - y1)
            rect = pygame.Rect(x1, y, wall_thickness, height)
        
        # TODO: handle collision for angled walls
        # else:
        #     # For non-axis-aligned walls, create a rect around the bounding box.
        #     x = min(x1, x2)
        #     y = min(y1, y2)
        #     width = abs(x2 - x1) or wall_thickness  # Ensure a nonzero width.
        #     height = abs(y2 - y1) or wall_thickness  # Ensure a nonzero height.
        #     rect = pygame.Rect(x, y, width, height)
        
        wall_rects.append(rect)
        pygame.draw.rect(screen, wall_color, rect)
    
    return wall_rects