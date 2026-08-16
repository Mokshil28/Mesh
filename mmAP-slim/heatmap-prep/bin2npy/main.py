from main_time_dop_read_all import get_time_angle, get_time_doppler, get_time_range
from split_spectrogram import split_spectrogram
from min_max_check import check_max_min_in_dir
import numpy as np
import os
import matplotlib.pyplot as plt


class FrameConfig:
    def __init__(self):
        self.numADCSamples=256 # number of data points per chirp
        self.numRxAntennas=4 # number of RX antennas
        # tx order tx0,tx2,tx1  face to the board (left,right,upper) 
        self.numTxAntennas=3 # number of TX antennas
        # num of chirp loop, one loop has three chirps
        self.numLoopsPerFrame=128 # number of chirps per frame per Tx antennas

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame

        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame
        self.numAngleBins = 64
      
        # calculate size of one chirp in short.
        self.chirpSize = self.numRxAntennas * self.numADCSamples 
        # calculate size of one chirp loop in short. 3Tx has three chirps in one loop for TDM.
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        # calculate size of one frame in short.
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame

class RawDataReader:
    def __init__(self, filename='./'):
        self.ADCBinFile=open(filename, 'rb')
       
    def getNextFrame(self,frameconfig):
        # read from binary file for one frame
        numpyFrame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize*4), dtype=np.int16)
        # this creates a numpy matrix filled with zero of the size of the frame
        numpyCompFrame=np.zeros(shape=(len(numpyFrame)//2), dtype=np.complex128)
        # this performs the transformation of the binary data to complex number matrix
        numpyCompFrame[0::2] = numpyFrame[0::4]+1j*numpyFrame[2::4]
        numpyCompFrame[1::2] = numpyFrame[1::4]+1j*numpyFrame[3::4] # transformations
        # try to learn what [1::2] means so that we can know what it is
        """according to my understanding it means to collect data 
        from a matrix that consist of [real1,real2,imaginary1,imaginary2,...] to something like this [real1+j*imaginary1, real2+1j*imaginary2, ....]"""
        return numpyCompFrame

    def close(self):
        self.ADCBinFile.close()

def frameReshape(frame, frameConfig):
    frameWithChirp = np.reshape(frame, (frameConfig.numLoopsPerFrame, frameConfig.numTxAntennas, frameConfig.numRxAntennas, -1))
    return frameWithChirp.transpose(1,2,0,3) # (tx, rx, no of chirps, no of samples) this is a matrix with 4 dimension
    # which each dimension being tx, rx , number of chirps and number of samples

def rangeFFT(data, frameConfig):    
    windowedBins1D = data*np.hamming(frameConfig.numADCSamples) # hamming window
    rangeFFTResult=np.fft.fft(windowedBins1D)
    return rangeFFTResult

if __name__ == '__main__':
    bin_directory = '/data/sxu7/heatmaps/recorded/bin/'
    
    for filename in os.listdir(bin_directory):
        
        activity = os.path.splitext(filename)[0]
        
        reader = RawDataReader(bin_directory + filename)
        frameConfig = FrameConfig()
        all_frames = []


        while True:
            frame = reader.getNextFrame(frameConfig)
            frame = frameReshape(frame,frameConfig)
            try:
                rangeFFT(frame,frameConfig)
            except Exception:
                break
            all_frames.append(frame)
            
        all_frames = np.stack(all_frames,axis = 0)
        #print(all_frames.shape)#debugging
        
        num_frames = all_frames.shape[0]
        train_size = int(num_frames * 0.8)
        frames_train = all_frames[:train_size]
        frames_eval = all_frames[train_size:]

        # #this generates the time doppler heatmaps
        # td = get_time_doppler(all_frames).cpu().numpy()
        # print('td',td.shape)#debugging    

        # #this generates the time range heatmaps
        # tr = get_time_range(all_frames).cpu().numpy()
        # tr = tr[:100,:]
        # print('tr',tr.shape)#debugging

        # #this generates the time angle heatmaps
        # ta = get_time_angle(all_frames).cpu().numpy()        
        # print('ta', ta.shape)#debugging
        
        ta_train = get_time_angle(frames_train).cpu().numpy()
        ta_eval = get_time_angle(frames_eval).cpu().numpy()
        print ('ta_train',ta_train.shape, ', ta_eval',ta_eval.shape)
        split_spectrogram(ta_train, True, activity, 'ta', filename)
        split_spectrogram(ta_eval, False, activity, 'ta', filename)
        
        td_train = get_time_doppler(frames_train).cpu().numpy()
        td_eval = get_time_doppler(frames_eval).cpu().numpy()
        print ('td_train',td_train.shape, ', td_eval',td_eval.shape)
        split_spectrogram(td_train, True, activity, 'td', filename)
        split_spectrogram(td_eval, False, activity, 'td', filename)
        
        tr_train = get_time_range(frames_train).cpu().numpy()
        tr_eval = get_time_range(frames_eval).cpu().numpy()
        print ('tr_train',tr_train.shape, ', tr_eval',tr_eval.shape)
        split_spectrogram(tr_train, True, activity, 'tr', filename)
        split_spectrogram(tr_eval, False, activity, 'tr', filename)
        
        # #this will crop the heatmaps in the desired shape
        # split_spectrogram(ta,'ta',filename)
        # split_spectrogram(td,'td',filename)
        # split_spectrogram(tr,'tr',filename)
        
    check_max_min_in_dir('/data/sxu7/heatmaps/recorded/finetune-cls/train/angle')
    check_max_min_in_dir('/data/sxu7/heatmaps/recorded/finetune-cls/train/doppler')
    check_max_min_in_dir('/data/sxu7/heatmaps/recorded/finetune-cls/train/range')
