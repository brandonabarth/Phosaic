import numpy as np


def gen_qm_map(image):
    qm_map = np.zeros(image.shape)
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if x-1 >= 0 and y-1 >= 0:
                left = qm_map[y][x-1]
                top = qm_map[y-1][x]
                corner = qm_map[y-1][x-1]
            elif x-1 >= 0 and y-1 < 0:
                left = qm_map[y][x-1]
                top = 0
                corner = 0
            elif x-1 < 0 and y-1 >= 0:
                left = 0
                top = qm_map[y-1][x]
                corner = 0
            else:
                left = 0
                top = 0
                corner = 0
                
            box_slice = corner
            row_slice = left - box_slice
            col_slice = top - box_slice
            
            qm_map[y][x] = image[y][x] + box_slice + row_slice + col_slice
    
    return qm_map
            
            
def quick_mean(qm_map, top_left, bottom_right):
    if top_left[0] == 0 and top_left[1] == 0:
        total = qm_map[bottom_right[1]][bottom_right[0]]
    elif top_left[0] == 0 and top_left[1] > 0:
        total = qm_map[bottom_right[1]][bottom_right[0]] - qm_map[top_left[1]-1][bottom_right[0]]
    elif top_left[0] > 0 and top_left[1] == 0:
        total = qm_map[bottom_right[1]][bottom_right[0]] - qm_map[bottom_right[1]][top_left[0]-1]
    else:
        total = qm_map[bottom_right[1]][bottom_right[0]] - ((qm_map[top_left[1]-1][bottom_right[0]]) + (qm_map[bottom_right[1]][top_left[0]-1])) + qm_map[top_left[1]-1][top_left[0]-1]
        
    
    num_pixels = ((bottom_right[0]+1)-top_left[0]) * ((bottom_right[1]+1)-top_left[1])
    return total / num_pixels
