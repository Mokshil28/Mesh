# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on BEiT, timm, DINO, DeiT and MAE-priv code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from utils import create_transform

from .data_constants import (IMAGE_TASKS, IMAGENET_DEFAULT_MEAN,
                             IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN,
                             IMAGENET_INCEPTION_STD)
from .dataset_folder import ImageFolder, MultiTaskImageFolder, DatasetFolder, MultiTaskDatasetFolder, default_loader

import cv2
def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

import matplotlib.pyplot as plt
def visualize_heatmap_comparison(original_data, transformed_data, title, save_path):
    """
    Visualize original and transformed heatmap data side by side and save the plot.
    Maintains the original shape of the data for plotting.
    """
    plt.figure(figsize=(20, 15))
    tasks = ['angle', 'doppler', 'range']
    
    for i, task in enumerate(tasks):
        if task in original_data and task in transformed_data:
            original_array = original_data[task]
            transformed_array = transformed_data[task]
            
            # Original data
            plt.subplot(2, 3, i+1)
            plt.imshow(original_array, cmap='viridis')  # Removed aspect='auto'
            plt.title(f'{title} - Original {task}\nShape: {original_array.shape}')
            plt.colorbar()
            
            # Transformed data
            plt.subplot(2, 3, i+4)
            plt.imshow(transformed_array, cmap='viridis')  # Removed aspect='auto'
            plt.title(f'{title} - Transformed {task}\nShape: {transformed_array.shape}')
            plt.colorbar()
            
            # Print some statistics
            print(f"{task} - Original: shape={original_array.shape}, min={original_array.min():.2f}, max={original_array.max():.2f}, mean={original_array.mean():.2f}")
            print(f"{task} - Transformed: shape={transformed_array.shape}, min={transformed_array.min():.2f}, max={transformed_array.max():.2f}, mean={transformed_array.mean():.2f}")
            
            # Check if shapes are the same
            if original_array.shape == transformed_array.shape:
                print(f"{task} - Shape unchanged. Value difference: {np.allclose(original_array, transformed_array)}")
            else:
                print(f"{task} - Shape changed from {original_array.shape} to {transformed_array.shape}")
        else:
            print(f"Warning: {task} data not found in one or both dictionaries")
    
    plt.tight_layout()
    # plt.savefig(save_path)
    plt.close()


class DataAugmentationForHeatmap:
    def __init__(self, args):
        self.noise_rate = None
        self.direction = None
        self.shifted_rows = None
        self.last_cropped_position = None
        self.group_crop_indices = {}

    def add_gaussian_noise(self, data):
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self.noise_rate = np.random.rand() * 0.05
        signal_mean = np.mean(np.abs(data))
        real_noise = np.random.normal(0, signal_mean * self.noise_rate, size=data.shape).astype(np.float32)

        return data + real_noise

    def pixel_row_shift(self, data):
        #NOTE exponential random distribution
        # probabilities = np.exp(-np.arange(0, 24))
        # probabilities /= probabilities.sum()
        # shift = np.random.choice(np.arange(0, 24), p=probabilities)

        #NOTE linear random distribution
        # probabilities = np.linspace(1, 0, 24)
        # probabilities /= probabilities.sum()
        # shift = np.random.choice(np.arange(1, 25), p=probabilities)

        #NOTE uniform random distribution
        shift = np.random.randint(1, 25)

        direction = np.random.choice([0, 1])

        if direction == 1 and shift > 0:
            shift_data = np.roll(data, shift, axis=0)
            self.direction = 'top2bottom'
            self.shift_added = True
        elif direction == 0 and shift > 0:
            shift_data = np.roll(data, -shift, axis=0)
            self.direction = 'bottom2top'
            self.shift_added = True
        else:
            shift_data = data
            self.direction = 'none'
            self.shift_added = False

        self.shifted_rows = shift
        return shift_data

    def crop_and_interpol(self, data, task_group_key):
        height, width = data.shape

        if task_group_key not in self.group_crop_indices:
            start_col = np.random.randint(0, width // 2)
            end_col = np.random.randint(width // 2 + 1, width)
            self.group_crop_indices[task_group_key] = (start_col, end_col)

        start_col, end_col = self.group_crop_indices[task_group_key]

        cropped_data = data[:, start_col:end_col]
        
        cropped_data_tensor = torch.tensor(cropped_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized_data_tensor = TF.resize(cropped_data_tensor, size=[height, width], interpolation=TF.InterpolationMode.BILINEAR)
        resized_data = resized_data_tensor.squeeze().numpy()

        self.last_cropped_position = (start_col, end_col)
        return resized_data

    def __call__(self, task_dict):
        # print("DataAugmentationForHeatmap is being called")
        
        # tasks_to_process = ['angle', 'doppler', 'range']
        tasks_to_process = ['angle', 'range']
        
        for task in tasks_to_process:
            if task in task_dict:
                data = np.array(task_dict[task])

                
                if np.random.rand() < 0.8:
                    data = self.add_gaussian_noise(data)

                if np.random.rand() < 0.8:
                    data = self.pixel_row_shift(data)

                if np.random.rand() < 0.8:
                    group_key = f'group_{task}'
                    data = self.crop_and_interpol(data, group_key)

                task_dict[task] = data

        return task_dict

    def __repr__(self):
        repr_str = '(DataAugmentationForHeatmap,\n'
        repr_str += '  noise_rate=%s,\n' % str(self.noise_rate)
        repr_str += '  direction=%s,\n' % str(self.direction)
        repr_str += '  shifted_rows=%s,\n' % str(self.shifted_rows)
        repr_str += '  last_cropped_position=%s\n' % str(self.last_cropped_position)  
        repr_str += ')'
        return repr_str


def build_multimae_pretraining_dataset(args):
    if 'angle' in args.all_domains or 'doppler' in args.all_domains or 'range' in args.all_domains:
        # Create transform object only if augmentation is enabled
        heatmap_transform = DataAugmentationForHeatmap(args) if args.heatmap_aug else None
        
        # Create the actual dataset with or without transform
        dataset = MultiTaskImageFolder(args.data_path, args.all_domains, transform=heatmap_transform)
        
        if heatmap_transform:
            num_samples_to_visualize = 3
            for i in range(num_samples_to_visualize):
                original_sample, _ = dataset[i]
                transformed_sample = heatmap_transform(original_sample.copy())
                
                visualize_heatmap_comparison(
                    original_sample,
                    transformed_sample,
                    f"Pretrain Sample {i} (augmented)",
                    f"finetune_dataaug/[pretrain]Sample{i}_augmented.png"
                )
        
        print(f"Data augmentation is {'enabled' if heatmap_transform else 'disabled'}")
        
        return dataset
    else:
        raise NotImplementedError()



def build_dataset(is_train, args):
    if args.data_set == "heatmap":
        root = args.data_path if is_train else args.eval_data_path
        
        transform = DataAugmentationForHeatmap(args) if is_train and args.heatmap_aug else None
        
        dataset = MultiTaskImageFolder(root=root, tasks=args.in_domains, transform=transform)
        
        nb_classes = args.nb_classes
        
        # Optional debug visualizations are slow on external drives; skip by default.
        if transform and getattr(args, "debug_aug", False):
            num_samples_to_visualize = 3
            for i in range(num_samples_to_visualize):
                original_sample, _ = dataset[i]
                transformed_sample = transform(original_sample.copy())
                
                visualize_heatmap_comparison(
                    original_sample,
                    transformed_sample,
                    f"Finetune Sample {i} (augmented)",
                    f"finetune_dataaug/[finetune]Sample{i}_augmented.png"
                )
        
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print(f"Number of classes = {args.nb_classes}")
    print(f"Data augmentation is {'enabled' if transform else 'disabled'}")

    return dataset, nb_classes



