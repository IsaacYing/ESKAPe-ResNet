"""
Bacteria Patch Extraction Pipeline
=================================
1. Load multi-channel TIFF microscopy image
2. Run Omnipose segmentation on brightfield channel
3. Crop each detected bacterium with padding & resize to 224x224
4. Pack into pickle batches for downstream model training

Input:  Raw TIFF file (e.g., Well-A01_file_name.tif)
Output: Batched pickle files (e.g., xxx_Batch_1.pkl)
"""

import os
import re
import random
import string
import datetime
import pickle as pk
import numpy as np
import cv2
import tifffile as tif
from skimage.measure import regionprops
from cellpose_omni import models as omni_models
from omnipose.utils import normalize99

random_string = generate_random_string()

class Vividict(dict):
    def __missing__(self,key):
        value = self[key] = type(self)()
        return value

def generate_random_string():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(3))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seg_model = omni_models.CellposeModel(gpu=True,model_type='bact_phase_omni')
params = {'channels':[0,0], # always define this with the model
          'rescale': 1.5, # upscale or downscale your images, None = no rescaling 
          'mask_threshold':0, # erode or dilate masks with higher or lower values between -5 and 5 
          'flow_threshold': 0, # default is .4, but only needed if there are spurious masks to clean up; slows down output
          'transparency': True, # transparency in flow output
          'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
          'cluster': True, # use DBSCAN clustering
          'resample': True, # whether or not to run dynamics on rescaled grid or original grid 
          'verbose': False, # turn on if you want to see more output 
          'tile': False, # average the outputs from flipped (augmented) images; slower, usually not needed 
          'niter': None, # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
          'augment': False, # Can optionally rotatethe image and average network outputs, usually not needed 
          'affinity_seg': False, # new feature, stay tuned...
         }

def prepare_bacteria_image(cell_img, target_size=224, 
                          padding_mode='reflect', output_dtype='float32'):
    if cell_img.ndim != 2:
        raise ValueError("The input must be a two-dimensional grayscale image")
    
    dtype_info = np.iinfo(cell_img.dtype) if cell_img.dtype.kind in 'ui' else np.finfo(cell_img.dtype)
    img_float = cell_img.astype(np.float32)

    h, w = cell_img.shape
    scale_factor = 1.0
    if target_size > max(h, w):
        scale = 1.0
    else: 
        scale = target_size / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img_float, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    padding_dict = {
        'reflect': cv2.BORDER_REFLECT_101,
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE
    }
    border_type = padding_dict.get(padding_mode, cv2.BORDER_REFLECT_101)
    
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=border_type,
    )
    
    return padded.astype(np.float32)

def record(content):
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M")
    with open(f"{random_string}_log.txt", "a") as f:
        f.write(time_str)
        f.write("\n")
        f.write(content)
        f.write("\n")

def well_pattern_extract(s):
    pattern = r'Well-([A-Z]\d{2})_'
    match = re.search(pattern, s)
    return match.group(1) if match else None
    

file_path = "path/file_name.tif"
well = well_pattern_extract(file_path)
save_dir = f"savedir/{random_string}_{well}/"
temp_string = f"{random_string}_{well}"

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File does not exist: {file_path}")

miniBatchSize = 1000
mini_batch = Vividict()
temp_batch = Vividict()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_name = os.path.basename(file_path)

img = tif.imread(file_path)
brightfield_img = img[0]

normalized_img = normalize99(brightfield_img)

masks, flows, styles = seg_model.eval(normalized_img, **params)

props = regionprops(masks)
prop_count = str(len(props))
record(prop_count)
if prop_count == 0:
    record(f"Warning: No cells detected in file {file_name}")


for cell_idx, prop in enumerate(props):
    # Note that Cid starts from 1
    cid = file_name + "_" + str(cell_idx)

    min_row, min_col, max_row, max_col = prop.bbox

    padding = 0
    y1 = max(0, min_row - padding)
    y2 = min(normalized_img.shape[0], max_row + padding)
    x1 = max(0, min_col - padding)
    x2 = min(normalized_img.shape[1], max_col + padding)
    
    cell_img = brightfield_img[y1:y2, x1:x2]
    pad_img = prepare_bacteria_image(cell_img)
    temp_batch[cid] = pad_img
    mini_batch[cid] = 0
    bac_count = len(mini_batch)
    is_batch_full = (bac_count % miniBatchSize == 0)
    is_last_cell = (cell_idx == len(props) - 1)
    if is_batch_full:
        batch_id = bac_count // miniBatchSize
        save_path = f"{save_dir}/{random_string}_Batch_{batch_id}.pkl"
        pk.dump(temp_batch, open(save_path, 'wb'))
        temp_batch = {}
    elif is_last_cell:
        batch_id = bac_count // miniBatchSize + 1
        save_path = f"{save_dir}/{random_string}_Batch_{batch_id}.pkl"
        pk.dump(temp_batch, open(save_path, 'wb'))
        temp_batch = {}
record("Finish")