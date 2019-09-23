## Matterport3d cube2sphere
# Prerequirements:
# 1. Download the "skybox" category of Matterport3d dataset.
# 2. Install cube2sphere library

import os
import zipfile

from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

matterport_dataset_path = './data/matterport3d/train/v1/scans/'
panorama_output_path = './data/matterport3d/cube2sphere_output/'
clipped_panorama_output_path = './data/matterport3d_panorama/'

cube2sphere_output_shape = (768, 384)
edge_cutoff_len = 64
final_shape = (768, 256) # (768, 384) => (768, 384-64*2) => (768, 256) => (64x12, 64x4)

if not os.path.exists(panorama_output_path):
    os.mkdir(panorama_output_path)
if not os.path.exists(clipped_panorama_output_path):
    os.mkdir(clipped_panorama_output_path)

# unzip downloaded data
def unzip(zip_path, to_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(to_path)
    zip_ref.close()

all_house_dirs = glob(matterport_dataset_path + '*')

all_panorama_imgs = []
all_panorama_ids = {} # retain a ```panoram_id->house_dir``` mapping

# Allocate all panorama images
for d in tqdm(all_house_dirs):
    
    house_id = d.split("/")[-1]
    
    # unzip file if hasn't yet
    if not os.path.exists(d + '/' + house_id):
        print("!", house_id)
        unzip(d + '/matterport_skybox_images.zip', to_path=d)
    
    skybox_data_path = "{}/{}/matterport_skybox_images/".format(d, house_id)
    for name in os.listdir(skybox_data_path):
        if 'jpg' in name:
            panorama_id = name[:-4].split('_')[0]
            all_panorama_ids[panorama_id] = skybox_data_path

skybox_idx = {
    'top': 0,
    'back': 1,
    'right': 2,
    'front': 3,
    'left': 4,
    'bottom': 5,
}

def compose_skybox_path(position, panorama_id, panorama_path):
    pos_idx = skybox_idx[position]
    return '{}/{}_skybox{}_sami.jpg'.format(panorama_path, panorama_id, pos_idx)

for panorama_id in tqdm(all_panorama_ids):
    path = all_panorama_ids[panorama_id]

    command = ' '.join([
        "cube2sphere",
        compose_skybox_path('front', panorama_id, path),
        compose_skybox_path('back', panorama_id, path),
        compose_skybox_path('right', panorama_id, path),
        compose_skybox_path('left', panorama_id, path),
        compose_skybox_path('top', panorama_id, path),
        compose_skybox_path('bottom', panorama_id, path),
        "-r {} {}".format(cube2sphere_output_shape[0], cube2sphere_output_shape[1]),
        "-f PNG",
        "-o {}/{}".format(panorama_output_path, panorama_id)])
    
    os.system(command)

def clip_img_margin(img, clip_len=64):
    HH, WW, _ = img.shape
    # Somehow already clipped
    if final_shape[0]==HH and final_shape[1]==WW:
        return img
    else:
        return img[clip_len:HH-clip_len, :, :]
    
for panorama_id in tqdm(all_panorama_ids):
    img = plt.imread('{}/{}0001.png'.format(panorama_output_path, panorama_id))
    clipped_img = clip_img_margin(img)
    output_path = '{}/{}.png'.format(clipped_panorama_output_path, panorama_id)
    plt.imsave(output_path, clipped_img)