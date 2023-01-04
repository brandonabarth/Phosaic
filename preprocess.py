import sys, os
import cv2
import numpy as np
import quick_mean
from multiprocessing import Pool


def process_image(x):
    idx, image_path, source_path, dest_path = x
    im = cv2.imread(source_path + '/' + image_path)
    #l, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))
    l, a, b, = cv2.split(im)
    
    l_map = quick_mean.gen_qm_map(l)
    a_map = quick_mean.gen_qm_map(a)
    b_map = quick_mean.gen_qm_map(b)
    
    qm_map = np.dstack((l_map, a_map, b_map))
    np.save(dest_path + f'/qm_maps/{str(idx)}', qm_map)
    
    cv2.imwrite(dest_path + f'/images/{str(idx)}.jpg', im)

def preprocess(source_path, dest_path):
    try:
        os.mkdir(dest_path)
    except:
        pass
    try:
        os.mkdir(dest_path + '/images')
    except:
        pass
    try:
        os.mkdir(dest_path + '/qm_maps')
    except:
        pass

    pool = Pool()

    image_paths = os.listdir(source_path)
    args = [(x, image_paths[x], source_path, dest_path) for x in range(len(image_paths))]
    pool.map(process_image, args)
    
        
        
def main():
    source_path = sys.argv[1]
    dest_path = sys.argv[2]
    
    preprocess(source_path, dest_path)
        
if __name__ == '__main__':
    main()