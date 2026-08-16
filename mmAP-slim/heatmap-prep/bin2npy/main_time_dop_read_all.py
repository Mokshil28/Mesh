import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
import numpy as np
import torch
import time
import pickle as pkl
import glob
import torchaudio.transforms
from tqdm import tqdm
import scipy
import sys
# gpu_device=torch.device("cuda:0")
gpu_device=torch.device("cpu")

flag_add_noise = True
noise_rate = 0.05
def clutter_removal_gpu(mat, axis=-2):  # axis on the chirp dimension
    return mat - mat.mean(dim=axis, keepdims=True)

def MTI_filter(Data_range):
    tx_rx_size, length_size,  chirp_loops, adc_samples = Data_range.shape
    Data_range = Data_range.reshape(tx_rx_size*length_size*chirp_loops, adc_samples).T

    Data_range=Data_range.detach().cpu().numpy()
    Data_range_MTI = np.zeros_like(Data_range, dtype="complex128")
    [b, a] = scipy.signal.butter(4, 0.0075, 'high')
    for k in range(len(Data_range_MTI)):
        Data_range_MTI[k, 0: Data_range.shape[1]] = scipy.signal.lfilter(b, a, Data_range[k, 0: Data_range.shape[1]])
    Data_range_MTI=Data_range_MTI.T.reshape(tx_rx_size, length_size, chirp_loops, adc_samples)
    return torch.from_numpy(Data_range_MTI).to(gpu_device)

def fft_with_window_gpu(mat, axis=-1, shift_flag=False):
    mat_shape = mat.shape
    dim = np.ones_like(mat_shape, dtype=int)
    dim[axis] = mat_shape[axis]
    wd = torch.hamming_window(mat_shape[axis], periodic=False, alpha=0.54, beta=0.46, dtype=torch.float32).reshape(
        tuple(dim)).to(gpu_device)
    result = torch.fft.fft(mat * wd, dim=axis)
    if shift_flag:
        result = torch.fft.fftshift(result, dim=axis)
    return result

def fft_with_padding(mat, dim, size, shift_flag=False):
    result=torch.fft.fft(mat, dim=dim, n=size)
    if shift_flag:
        result=torch.fft.fftshift(result, dim=dim)
    return result

def get_time_doppler(data):
    length_size, tx_size, rx_size, chirp_loops, adc_samples = data.shape
    with torch.no_grad():
        data = torch.from_numpy(data).cfloat().to(gpu_device)
        if flag_add_noise:
            signal_mean = torch.mean(torch.abs(data))
            data += (torch.normal(0, signal_mean * noise_rate, size=data.shape)
                      + 1j * torch.normal(0, signal_mean * noise_rate, size=data.shape)).to(gpu_device)
        data = data.view(length_size, tx_size, rx_size, chirp_loops, adc_samples)
        data = torch.permute(data, (1, 2, 0, 3, 4)).view(tx_size * rx_size, length_size, chirp_loops, adc_samples)[0:1]


        frames_tensor = data
        frames_range = fft_with_window_gpu(frames_tensor, axis=-1, shift_flag=False)
        ##for each range, calculate stft and then sum
        frames_range = clutter_removal_gpu(frames_range, axis=-2)[...,0:150]#100
        # frames_range = MTI_filter(frames_range)

        frames_range = frames_range  # [...,30:100]
        stft_sum = 0
        transform = torchaudio.transforms.Spectrogram(n_fft=256, win_length=256, window_fn=torch.hamming_window,
                                                      hop_length=16, power=None,
                                                      center=False, onesided=False).to(gpu_device)
        for antenna in range(frames_range.shape[0]):
            for ran in range(frames_range.shape[-1]):
                time_doppler = torch.fft.fftshift(transform(frames_range[antenna, ..., ran].reshape(-1)), dim=0)
                time_doppler = torch.abs(time_doppler)
                stft_sum += time_doppler
        time_doppler = 20 * torch.log10(stft_sum)
        # print(time_doppler.shape)
        return time_doppler

def get_time_range(data):
    length_size, tx_size, rx_size, chirp_loops, adc_samples = data.shape

    with torch.no_grad():
        data = torch.from_numpy(data).cfloat().to(gpu_device)
        if flag_add_noise:
            signal_mean = torch.mean(torch.abs(data))
            data += (torch.normal(0, signal_mean * noise_rate, size=data.shape)
                      + 1j * torch.normal(0, signal_mean * noise_rate, size=data.shape)).to(gpu_device)
        data = data.view(length_size, tx_size, rx_size, chirp_loops, adc_samples)
        data = torch.permute(data, (1, 2, 0, 3, 4)).view(tx_size * rx_size, length_size, chirp_loops, adc_samples)[0:1]

        frames_tensor = data
        frames_range = fft_with_window_gpu(frames_tensor, axis=-1, shift_flag=False)
        ##for each range, calculate stft and then sum
        frames_range = clutter_removal_gpu(frames_range, axis=-2)[...,0:150]
        # frames_range = MTI_filter(frames_range)

        frames_range = frames_range.reshape(frames_range.shape[0], length_size*chirp_loops, -1)

        win_length = 256
        hop_length=16
        time_range_list=[]
        for time_start in range(0, frames_range.shape[-2] - win_length+1, hop_length):
            time_range_i = frames_range[:, time_start:time_start+win_length, :]
            time_range = torch.sum(torch.abs(time_range_i), dim=(0, 1))
            time_range_list.append(time_range)
        time_range=torch.stack(time_range_list, dim=0).T

        time_range = 20 * torch.log10(time_range)
        # print(time_doppler.shape)
        return time_range


def get_time_angle(data):
    length_size, tx_size, rx_size, chirp_loops, adc_samples = data.shape

    with torch.no_grad():
        data = torch.from_numpy(data).cfloat().to(gpu_device)
        if flag_add_noise:
            signal_mean = torch.mean(torch.abs(data))
            data += (torch.normal(0, signal_mean * noise_rate, size=data.shape)
                      + 1j * torch.normal(0, signal_mean * noise_rate, size=data.shape)).to(gpu_device)
        data = data.view(length_size, tx_size, rx_size, chirp_loops, adc_samples)
        data = torch.permute(data, (1, 2, 0, 3, 4)).view(tx_size * rx_size, length_size, chirp_loops, adc_samples)

        frames_tensor = data
        frames_range = fft_with_window_gpu(frames_tensor, axis=-1, shift_flag=False)
        ##for each range, calculate stft and then sum
        frames_range = clutter_removal_gpu(frames_range, axis=-2)[...,0:150]
        # frames_range = MTI_filter(frames_range)

        ANGLE_BIN_SIZE=128
        NUM_TX_LOWER=2
        NUM_RX=4
        antenna_index=[0,1,2,3,8,9,10,11]###only for sim
        frames_range_h = frames_range[antenna_index].reshape(NUM_TX_LOWER * NUM_RX, length_size * chirp_loops, -1)
        # frames_range_h = frames_range[:NUM_TX_LOWER*NUM_RX].reshape(NUM_TX_LOWER*NUM_RX, length_size*chirp_loops, -1)
        angle_h = fft_with_padding(frames_range_h, dim=0, size=ANGLE_BIN_SIZE, shift_flag=True)#ANGLE_BIN_SIZE, length_size*chirp_loops, adc_samples
        win_length = 256
        hop_length = 16
        time_angle_list=[]
        for time_start in range(0, frames_range_h.shape[-2] - win_length+1, hop_length):
            time_angle_i = angle_h[:, time_start:time_start+win_length, :]
            time_angle = torch.sum(torch.abs(time_angle_i), dim=(1, 2))
            time_angle_list.append(time_angle)
        time_angle=torch.stack(time_angle_list, dim=0).T
        time_angle = 20 * torch.log10(time_angle)
        # print(time_doppler.shape)
        return time_angle

if __name__=='__main__':
    save_folder = "data_noise_0.05_sigma_0.6"
    if not os.path.isdir("./exp2-3/{}".format(save_folder)):
        os.mkdir("./exp2-3/{}".format(save_folder))

    flag_add_noise = True
    noise_rate = 0.05


    trial_list = []
    trial_list.extend(["hongfei_a{}".format(x) for x in range(1, 9)])

    # trial_list .extend(["s20{:02d}".format(x) for x in [1,2,3,4,5,6,7,8,9,12,13]])
    # trial_list.extend(["s20{:02d}".format(x) for x in [38,39,40,41,42,43,44,45,46,47,49]])
    # trial_list.extend(["s20{:02d}".format(x) for x in [79,80,81,82,83,88,89,90,91,92]])
    # trial_list.extend(["s21{:02d}".format(x) for x in [1,2,3,4,5,6,7,8,9]])
    # trial_list.extend(["s21{:02d}".format(x) for x in [26,27,28,29,30,33,34,35,36,37]])


    total_time_begin = time.time()

    root_loc = "./mmMeshSim/data/simulator/signal_pytorch_i384_vspeed_sigma_0.6"
    for trial in trial_list:
        # outfolder = "./exp2-3/data_with_noise_0.05/{}_{}".format(trial, type)
        outfolder = "./exp2-3/{}/{}_sim".format(save_folder, trial)
        if not os.path.isdir(outfolder):
            os.mkdir(outfolder)
        data = np.load("{}/{}.dat".format(root_loc, trial))

        td = get_time_doppler(np.asarray(data)).cpu().numpy()
        np.save("{}/time_dop.npy".format(outfolder), td)

        tr = get_time_range(np.asarray(data)).cpu().numpy()
        np.save("{}/time_range.npy".format(outfolder), tr)

        ta = get_time_angle(np.asarray(data)).cpu().numpy()
        np.save("{}/time_angle.npy".format(outfolder), ta)

        cur_time = time.time()
        print("{} cost :".format(trial), cur_time - total_time_begin)