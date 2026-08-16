from main_time_dop_read_all import get_time_angle, get_time_doppler, get_time_range
from split_spectrogram import split_spectrogram
import numpy as np
import os
import torch
import torch.nn as nn

class FrameConfig:
    def __init__(self):
        self.numADCSamples = 256  # number of data points per chirp
        self.numRxAntennas = 4  # number of RX antennas
        self.numTxAntennas = 3  # number of TX antennas
        self.numLoopsPerFrame = 128  # number of chirps per frame per Tx antennas

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame
        self.numAngleBins = 64

        self.chirpSize = self.numRxAntennas * self.numADCSamples
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame

class RawDataReader:
    def __init__(self, filename='./'):
        self.ADCBinFile = open(filename, 'rb')

    def getNextFrame(self, frameconfig):
        numpyFrame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize * 4), dtype=np.int16)
        numpyCompFrame = np.zeros(shape=(len(numpyFrame) // 2), dtype=np.complex128)
        numpyCompFrame[0::2] = numpyFrame[0::4] + 1j * numpyFrame[2::4]
        numpyCompFrame[1::2] = numpyFrame[1::4] + 1j * numpyFrame[3::4]
        return numpyCompFrame

    def close(self):
        self.ADCBinFile.close()

def frameReshape(frame, frameConfig):
    frameWithChirp = np.reshape(frame, (frameConfig.numLoopsPerFrame, frameConfig.numTxAntennas, frameConfig.numRxAntennas, -1))
    return frameWithChirp.transpose(1, 2, 0, 3)

def rangeFFT(data, frameConfig):
    windowedBins1D = data * np.hamming(frameConfig.numADCSamples)
    rangeFFTResult = np.fft.fft(windowedBins1D)
    return rangeFFTResult

class HeatmapGenerator(nn.Module):
    def __init__(self):
        super(HeatmapGenerator, self).__init__()

    def forward(self, all_frames):
        td = get_time_doppler(all_frames)
        tr = get_time_range(all_frames)
        ta = get_time_angle(all_frames)
        return td, tr, ta

if __name__ == '__main__':
    bin_directory = '/data/sxu7/heatmaps/recorded-bin/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and wrap it in DataParallel with both GPUs
    model = HeatmapGenerator()
    model = nn.DataParallel(model, device_ids=[0, 1])  # Use both GPUs 0 and 1
    model.to(device)

    for filename in os.listdir(bin_directory):
        reader = RawDataReader(bin_directory + filename)
        frameConfig = FrameConfig()
        all_frames = []

        while True:
            frame = reader.getNextFrame(frameConfig)
            if len(frame) == 0:
                break
            frame = frameReshape(frame, frameConfig)
            try:
                rangeFFT(frame, frameConfig)
            except Exception:
                break
            all_frames.append(frame)

        all_frames = np.stack(all_frames, axis=0)
        all_frames = torch.tensor(all_frames, dtype=torch.complex128).to(device)

        # Forward pass through the model
        td, tr, ta = model(all_frames)

        # Move data back to CPU and convert to numpy
        td = td.cpu().numpy()
        tr = tr.cpu().numpy()
        ta = ta.cpu().numpy()

        # Process heatmaps
        split_spectrogram(ta, 'ta', filename)
        split_spectrogram(td, 'td', filename)
        split_spectrogram(tr, 'tr', filename)

        reader.close()
