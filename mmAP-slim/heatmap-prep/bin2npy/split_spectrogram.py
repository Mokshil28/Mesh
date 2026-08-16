import os
import numpy as np


def split_spectrogram(spectrogram_frame, is_train, activity, heatmap_type, filename):

    base_dir = '/data/sxu7/heatmaps/recorded'
    if is_train:
        train_eval = 'train'
    else:
        train_eval = 'eval'
                
    angle_dir = f'{base_dir}/finetune-cls/{train_eval}/angle'
    doppler_dir = f'{base_dir}/finetune-cls/{train_eval}/doppler'
    range_dir = f'{base_dir}/finetune-cls/{train_eval}/range'

    if heatmap_type == 'ta':
        save_dir = angle_dir
    elif heatmap_type == 'td':
        save_dir = doppler_dir
    elif heatmap_type == 'tr':
        save_dir = range_dir

    os.makedirs(save_dir, exist_ok=True)
    
    shift = 16
    for i in range(128,spectrogram_frame.shape[1],shift):
        spectrogram_frame_small = spectrogram_frame[:,i-128:i]
        
        os.makedirs(f'{save_dir}/{activity}', exist_ok=True)
        save_filename = f"{save_dir}/{activity}/{filename.split('.')[0]}_{heatmap_type}{str(i//shift + 1)}.npy"
        
        np.save(save_filename,spectrogram_frame_small)

        #print(save_filename)#debugging
        
    print (f'{activity} {train_eval} {heatmap_type} saved')