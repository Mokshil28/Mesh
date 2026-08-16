import os
import numpy as np

def check_max_min_in_dir(directory):
    overall_max = -np.inf
    overall_min = np.inf
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".npy"):
                filepath = os.path.join(root, filename)
                array = np.load(filepath)
                file_max = np.max(array)
                file_min = np.min(array)
                overall_max = max(overall_max, file_max)
                overall_min = min(overall_min, file_min)
                
                relative_path = os.path.relpath(filepath, directory)
                # print(f"File: {relative_path}")
                # print(f"Max value: {file_max}")
                # print(f"Min value: {file_min}\n")
    
    print(f"Overall Max value in {directory}: {overall_max}")
    print(f"Overall Min value in {directory}: {overall_min}\n")

ta_dir = '/data/sxu7/heatmaps/recorded/finetune-cls/train/angle'
td_dir = '/data/sxu7/heatmaps/recorded/finetune-cls/train/doppler'
tr_dir = '/data/sxu7/heatmaps/recorded/finetune-cls/train/range'

check_max_min_in_dir(ta_dir)
check_max_min_in_dir(td_dir)
check_max_min_in_dir(tr_dir)
