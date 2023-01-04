import cv2
import numpy as np


def rectangular_tiler(target_image, params):
    final_size = params['final_size']
    tile_size = params['tile_size']
    final_pixels = final_size[0]*tile_size[0], final_size[1]*tile_size[1]

    height, width = target_image.shape[0], target_image.shape[1]
    if width/height > (final_pixels[0] / final_pixels[1]): height_constrained = True
    else: height_constrained = False
    
    if height_constrained:
        top = 0
        bottom = height
        crop_width = height * (final_pixels[0]/final_pixels[1])
        left = (width - crop_width)/2
        right = left + crop_width
    else:
        left = 0
        right = width
        crop_height = width * (final_pixels[1]/final_pixels[0])
        top = (height - crop_height)/2
        bottom = top + crop_height
    
    crop_target_image = target_image[int(top):int(bottom), int(left):int(right)]
    big_target_image = cv2.resize(crop_target_image, dsize=final_pixels)
    
    #create tiles
    target_tiles = []
    mask = np.ones
    size = tile_size
    
    for x in range(final_size[0]):
        for y in range(final_size[1]):
            location = x*tile_size[0], y*tile_size[1]
            image = big_target_image[location[1]:location[1]+size[1], location[0]:location[0]+size[0]]
            #mean = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).mean(axis=(0,1))
            mean = image.mean(axis=(0,1))
            std = image.std(axis=(0,1)).mean()
            target_tiles.append({
                'mask': mask,
                'size': size,
                'location': location,
                'image': image,
                'mean': mean,
                'std': std
            })
            
    return target_tiles, final_pixels