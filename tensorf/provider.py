import os
import cv2
import glob
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from scipy.spatial.transform import Slerp, Rotation

from nerf.utils import *


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


class NeRFDataset(Dataset):
    def __init__(self, path, type='train', mode='colmap', preload=False, downscale=1, scale=0.33, n_test=10):
        super().__init__()
        # path: the json file path.

        self.root_path = path
        self.type = type # train, val, test
        self.mode = mode # colmap, blender, llff
        self.downscale = downscale
        self.preload = preload # preload data into GPU

        # camera radius scale to make sure camera are inside the bounding box.
        self.scale = scale

        # load nerf-compatible format data.
        if mode == 'colmap':
            with open(os.path.join(path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # only load one specified split
            else:
                with open(os.path.join(path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])    
        
        # for colmap, manually interpolate a test set.
        if mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all': use all frames
            
            self.poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if mode == 'blender':
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = np.stack(self.poses, axis=0)
        if self.images is not None:
            self.images = np.stack(self.images, axis=0)

        # load intrinsics
        
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('cannot read focal!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)

        # pre-generate rays from poses
        self.directions = get_ray_directions(self.H, self.W, [fl_x, fl_y], [cx, cy])  # [H, W, 3]

        self.all_rays_o = []
        self.all_rays_d = []
        for pose in self.poses:
            pose = torch.from_numpy(pose)
            rays_o, rays_d = get_rays(self.directions, pose)  # [H, W, 3]
            self.all_rays_o.append(rays_o)
            self.all_rays_d.append(rays_d)
            
        if self.images is not None:
            self.all_rgbs = []
            for image in self.images:
                rgb = torch.from_numpy(image) # rgb(a), [H, W, 3/4]
                self.all_rgbs.append(rgb)
        else:
            self.all_rgbs = None

        # free 
        del self.images
        del self.poses
        
        # stack
        self.all_rays_o = torch.stack(self.all_rays_o, dim=0) # [N, H, W, 3]
        self.all_rays_d = torch.stack(self.all_rays_d, dim=0) # [N, H, W, 3]
        if self.all_rgbs is not None:
            self.all_rgbs = torch.stack(self.all_rgbs, dim=0)

        # mix all rays from different images in training
        if self.type != 'test':
            self.all_rays_o = self.all_rays_o.view(-1, 3) # [NHW, 3]
            self.all_rays_d = self.all_rays_d.view(-1, 3) # [NHW, 3]
            if self.all_rgbs is not None:
                self.all_rgbs = self.all_rgbs.view(-1, self.all_rgbs.shape[-1])
        
        if preload:
            self.all_rays_o = self.all_rays_o.cuda()
            self.all_rays_d = self.all_rays_d.cuda()
            if self.all_rgbs is not None:
                self.all_rgbs = self.all_rgbs.cuda()


    def __len__(self):
        return self.all_rays_o.shape[0]

    def __getitem__(self, index):

        results = {
            'rays_o': self.all_rays_o[index],
            'rays_d': self.all_rays_d[index],
            'index': index,
        }

        if self.type != 'test' or self.mode == 'blender':    
            results['rgbs'] = self.all_rgbs[index]
        
        return results