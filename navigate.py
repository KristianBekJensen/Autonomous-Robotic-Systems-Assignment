import pygame

def navigate(keys, v_left, v_right):
    if keys[pygame.K_w]:
        if v_left != v_right:
            v_right = max(v_right, v_left)
            v_left = max(v_right, v_left)
        else:
            v_right += 0.2
            v_left += 0.2
    elif keys[pygame.K_s]:
        v_left = max(0, v_left - 0.2)
        v_right = max(0, v_right - 0.2)
    elif keys[pygame.K_d]:
        v_right *= 0.99
    elif keys[pygame.K_a]:
        v_left *= 0.99
    elif keys[pygame.K_e]:
        if v_left == 0 and v_right == 0:
            v_left = 0.3
            v_right = -v_left
    elif keys[pygame.K_q]:
        if v_left == 0 and v_right == 0:
            v_right = 0.3
            v_left = -v_right
    else:
        if v_right < 0 or v_left < 0:
            v_right = v_left = 0
        elif v_right >= v_left:
            v_left = v_right
        else:
            v_right = v_left
    return v_left, v_right