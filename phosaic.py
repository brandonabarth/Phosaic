import sys, os
from p_tqdm import p_map
import cv2
import numpy as np
import cupy as cp
import math
from cupyx.scipy.signal import correlate2d
from scipy.signal import correlate2d as correlate2dCPU
from scipy.spatial.distance import cdist
from tqdm import tqdm
import tilers
from quick_mean import quick_mean

import matplotlib.pyplot as plt



class ImageSet:
    def __init__(self, source_path=None):
        self.images = []
        self.qm_maps = []
        
        if source_path:
            self.load_from_dir(source_path)
            
    def load_from_dir(self, source_path):
        NUM_TO_LOAD = 6160
        print('loading image set from directory')
        print('loading images...')
        image_files = os.listdir(source_path + '/images')
        for file in tqdm(image_files[:NUM_TO_LOAD]):
            image = cv2.imread(source_path+'/images/'+file)
            self.images.append(image)
            
        print('loading qm_maps...')
        qm_map_files = os.listdir(source_path + '/qm_maps')
        for file in tqdm(qm_map_files[:NUM_TO_LOAD]):
            self.qm_maps.append(np.load(source_path+'/qm_maps/'+file))
        


class Phosaic:
    def __init__(self, image_set, final_tiles, final_size):
        self.image_set = image_set
        self.final_tiles = final_tiles
        self.final_size = final_size
        
    def generate_phosaic(self, outpath):
        self.final_image = np.zeros((self.final_size[1], self.final_size[0], 3))
        
        for tile in self.final_tiles:
            loc_x, loc_y = tile['location']
            tile_x, tile_y = tile['size']
            self.final_image[loc_y:loc_y+tile_y, loc_x:loc_x+tile_x, :] = tile['processed_image']
            
        cv2.imwrite(outpath, self.final_image)
        
        
        
class PhosaicProject:
    @staticmethod
    def process_corr_map(args):
        tile_size, gs_tile_size, image_size, gs_image_size, corr_maps = args
        
        inv_scale_factor = (image_size[0] / gs_image_size[0], image_size[1] / gs_image_size[1])
        perfect_scaling_tile_size = (gs_tile_size[0]*inv_scale_factor[0], gs_tile_size[1]*inv_scale_factor[1])
        
        max_corrs = []
        max_locs = []
        for corr_map in corr_maps:
            max_corr_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
            max_corr = corr_map[max_corr_idx]
            
            top_left = (max_corr_idx[1]*inv_scale_factor[0], max_corr_idx[0]*inv_scale_factor[1])
            center = (top_left[0]+(perfect_scaling_tile_size[0]/2), top_left[1]+(perfect_scaling_tile_size[1]/2))
            loc = [int(center[0]-(tile_size[0]/2)), int(center[1]-(tile_size[1]/2))]
            
            if loc[0] < 0: loc[0] = 0
            if loc[1] < 0: loc[1] = 0
            if loc[0] > image_size[0] - tile_size[0]: loc[0] = image_size[0] - tile_size[0]
            if loc[1] > image_size[1] - tile_size[1]: loc[1] = image_size[1] - tile_size[1]
            
            max_locs.append(loc)
            max_corrs.append(max_corr)
            
        return(max_locs, max_corrs)

    def do_tiling(self, tiler, target_image):
        self.target_tiles, self.final_size = tiler['func'](target_image, tiler['params'])
        
    def calc_corr_map(self, tile_idx):
        tile = self.target_tiles[tile_idx]
        gs_tile = tile['gs_tile']
        
        #do correlation calculations on GPU
        corr_maps = []
        for gs_image in self.image_set.gs_images:
            #print(gs_tile.max(), gs_tile.min())
            #print(gs_image.max(), gs_image.min())
            corr = correlate2d(gs_image, gs_tile, mode='valid')
            #print(corr)
            #print()
            corr_maps.append(corr)
            
        tile['corr_maps'] = corr_maps
        
    def calc_color_dists(self, tile_idx):
        tile = self.target_tiles[tile_idx]
        size = tile['size']
        
        #avg_colors = np.zeros((len(self.image_set.images), 3))
        avg_colors = []
        for img_idx in range(len(self.image_set.images)):
            b_map = self.image_set.qm_maps[img_idx][:, :, 0]
            g_map = self.image_set.qm_maps[img_idx][:, :, 1]
            r_map = self.image_set.qm_maps[img_idx][:, :, 2]
            
            loc = tile['max_locs'][img_idx]
            
            b = quick_mean(b_map, loc, (loc[0]+size[0]-1, loc[1]+size[1]-1))
            g = quick_mean(g_map, loc, (loc[0]+size[0]-1, loc[1]+size[1]-1))
            r = quick_mean(r_map, loc, (loc[0]+size[0]-1, loc[1]+size[1]-1))
            
            avg_colors.append([b, g, r])
            
        tile_avg = np.array([[tile['mean'][0], tile['mean'][1], tile['mean'][2]]])
        #tile_avg = tile['mean']
        color_dists = cdist(np.asarray(avg_colors), tile_avg)
        tile['color_dists'] = [x[0] for x in color_dists]
       

    def calc_tile_match_values(self):
        #create scaled down grayscale images and send to GPU
        print('processing images...')
        scale_factor = self.params['scale_down_factor']
        self.image_set.gs_images = []
        for image in tqdm(self.image_set.images):
            gs_image = cv2.cvtColor(cv2.resize(image, (int(image.shape[1]*scale_factor), int(image.shape[0]*scale_factor))), cv2.COLOR_BGR2GRAY)
            
            gs_mean = gs_image.mean(axis=(0,1))
            gs_std = gs_image.std(axis=(0,1))
            #if gs_std <= 0: gs_std = 0.01
            
            gs_image = (gs_image - gs_mean) / gs_std
            self.image_set.gs_images.append(cp.asarray(gs_image))
            
        #create scaled down grayscale tiles and send to GPU
        print('processing tiles...')
        for tile in tqdm(self.target_tiles):
            gs_tile = cv2.cvtColor(cv2.resize(tile['image'], (int(tile['size'][0]*scale_factor), int(tile['size'][1]*scale_factor))), cv2.COLOR_BGR2GRAY)
            
            gs_mean = gs_tile.mean(axis=(0,1))
            gs_std = gs_tile.std(axis=(0,1))
            #if gs_std <= 0: gs_std = 0.01
            
            gs_tile = (gs_tile - gs_mean) / (gs_std * gs_tile.shape[1] * gs_tile.shape[0])
            tile['gs_tile'] = cp.asarray(gs_tile)
            
        #calculate correlation maps
        print('calculating correlation maps...')
        for tile_idx in tqdm(range(len(self.target_tiles))):
            self.calc_corr_map(tile_idx)
        
       
        #process correlation maps
        print('getting correlation maps from GPU...')
        image_size = (self.image_set.images[0].shape[1], self.image_set.images[0].shape[0])
        gs_image_size = (self.image_set.gs_images[0].shape[1], self.image_set.gs_images[0].shape[0])
        
        args = []
        for tile in tqdm(self.target_tiles):
            corr_maps = [corr_map.get() for corr_map in tile['corr_maps']]
        
            tile_size = tile['size']
            gs_tile_size = (tile['gs_tile'].shape[1], tile['gs_tile'].shape[0])
            
            args.append((tile_size, gs_tile_size, image_size, gs_image_size, corr_maps))
        
        print('processing correlation maps')
        processed_corr_maps = p_map(self.process_corr_map, args)
        for tile_idx in range(len(self.target_tiles)):
            tile = self.target_tiles[tile_idx]
            tile['max_locs'] = processed_corr_maps[tile_idx][0]
            tile['max_corrs'] = processed_corr_maps[tile_idx][1]
            
        #get average color for each crop of highest correlation
        print('calculating color distance...')
        for tile_idx in tqdm(range(len(self.target_tiles))):
            self.calc_color_dists(tile_idx)
        
        
    def place_tile(self, tile_idx):
        tile = self.target_tiles[tile_idx]
    
        #normalize color and correlation values
        corr = np.asarray(tile['max_corrs']).astype('float64')
        corr = corr - corr.min()
        corr = (corr * 255.0) / corr.max()
        
        color_dist = np.ravel(tile['color_dists']).astype('float64')
        color_dist = color_dist - color_dist.min()
        color_dist = 255.0 - ((color_dist * 255.0) / color_dist.max())
        
        #do radius check
        location = tile['location']
        out_of_radius = np.ones(len(self.image_set.images))
        for image_idx in range(len(self.image_set.images)):
            for point in self.image_placements[image_idx]:
                dist = math.sqrt((point[0]-location[0])**2 + (point[1]-location[1])**2)
                if dist < self.params['min_radius']:
                    out_of_radius[image_idx] = 0
        
        #get high score
        score = (color_dist + corr*(tile['std']/50)) * out_of_radius
        high_score_idx = np.argmax(score)
        
        #update final tile
        tile_x, tile_y = tile['size']
        loc_x, loc_y = tile['max_locs'][high_score_idx]
        final_image = self.image_set.images[high_score_idx][loc_y:loc_y+tile_y, loc_x:loc_x+tile_x]
        self.final_tiles[tile_idx] = {
            'image': final_image,
            'mask': tile['mask'],
            'location': tile['location'],
            'size': (tile_x, tile_y)
        }
        
        #update location matrix
        self.image_placements[high_score_idx].append(tile['location'])
        
    def postprocess(self, tile_idx):
        tile = self.target_tiles[tile_idx]
        final_tile = self.final_tiles[tile_idx]
        color_shift_percent = self.params['color_shift_percent']
        std_shift_percent = self.params['std_shift_percent']
        
        tile_image_lab = cv2.cvtColor(tile['image'], cv2.COLOR_BGR2LAB).astype(np.float64)
        final_image_lab = cv2.cvtColor(final_tile['image'], cv2.COLOR_BGR2LAB).astype(np.float64)
        
        tile_mean = tile_image_lab.mean(axis=(0,1))
        tile_std = tile_image_lab.std(axis=(0,1))
        final_mean = final_image_lab.mean(axis=(0,1))
        final_std = final_image_lab.std(axis=(0,1))
        
        final_std[final_std <= 0] = 1
        
        tile_mean = np.clip(tile_mean, final_mean - (255*color_shift_percent), final_mean + (255*color_shift_percent))
        tile_std = np.clip(tile_std, (1-std_shift_percent)*final_std, (1+std_shift_percent)*final_std)
        
        transfer_image_lab = (final_image_lab - final_mean) / final_std
        transfer_image_lab = (transfer_image_lab * tile_std) + tile_mean
        transfer_image_lab = np.clip(transfer_image_lab, 0, 255)
        
        final_tile['processed_image'] = cv2.cvtColor(transfer_image_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        

    def __init__(self, image_set, target_image, tiler, params):
        self.image_set = image_set
        self.target_image = target_image
        self.params = params
    
        print('generating tiles...')
        #create target tiles
        self.do_tiling(tiler, target_image) 
        self.final_tiles = [None for x in range(len(self.target_tiles))]
        
        #calculate tile match values
        self.calc_tile_match_values()
        
        #assemble mosaic
        print('assembling mosaic...')
        self.image_placements = [[] for x in range(len(self.image_set.images))]
        for tile_idx in tqdm(range(len(self.target_tiles))):
            self.place_tile(tile_idx)
            
        #do post-processing
        if self.params['do_postprocessing']:
            print('doing postprocessing...')
            for tile_idx in tqdm(range(len(self.target_tiles))):
                self.postprocess(tile_idx)
            
            
    def get_phosaic(self):
        return Phosaic(self.image_set, self.final_tiles, self.final_size)
        
        

def main():
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    final_path = sys.argv[3]
    target_image = cv2.imread(target_path)
    
    image_set = ImageSet(source_path)
    
    tiler = {
        'func': tilers.rectangular_tiler,
        'params': {
            'final_size': (16, 40),
            'tile_size': (384, 216)
        }
    }
    
    params = {
        'scale_down_factor': .1,
        'min_radius': 5000,
        'do_postprocessing': True,
        'color_shift_percent': 0.1,
        'std_shift_percent': 0.1
    }
    
    new_proj = PhosaicProject(image_set, target_image, tiler, params)
    new_phosaic = new_proj.get_phosaic()
    new_phosaic.generate_phosaic(final_path)

          
if __name__ == '__main__':
    main()