import ESN
import torch
import wfdb
import numpy as np
createVar = locals()
import neurokit2 as nk
import utility
import statistics
from utility import random_index_generator

class timeseries_loader():
    def __init__(self,random_list = None,num_series:int = 10,validation:bool=True,length:int=200000,num_select_seg:int=100,flipping:bool=True,training_ratio:float=0.8, validation_ratio:float = 0.1, testing_ratio:float=0.1,peakwindow = 0.261, beatwindow = 0.767):
        self.max_length = 0
        self.min_length = 0
        self.abs_large_value = 0
        self.num_series = num_series
        self.validation = validation
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.testing_ratio = testing_ratio
        self.length = length
        self.flipping = flipping
        self.num_select_seg = num_select_seg
        self.random_list = []
        self.washout = 20000
        self.peakwindow = peakwindow
        self.beatwindow = beatwindow
        if random_list is None:
            random_list = random_index_generator(num_index=num_series)
        self._loading(random_list)
        self._peakfinding()
        self._segment_decision()
        self._preprocessing()
        if self.flipping is True:
            self._flipping()
        self._padding()
        self._normalization()
        self._dataset_partition()

    def _loading(self, random_list):
        self.random_list = random_list
        for i in range(0, self.num_series):
            if self.random_list[i] < 10:
                file_num = "000" + str(self.random_list[i])
            elif self.random_list[i] >= 10 and self.random_list[i] < 100:
                file_num = "00" + str(self.random_list[i])
            elif self.random_list[i] >= 100 and self.random_list[i] < 1000:
                file_num = "0" + str(self.random_list[i])
            else:
                file_num = str(self.random_list[i])
            try:
                createVar['origin_p{}'.format(i)] = wfdb.rdrecord('./data/' + file_num, channels=[2])
                createVar['origin_p{}'.format(i)] = np.squeeze(
                    createVar['origin_p{}'.format(i)].p_signal[self.washout:self.washout + self.length])
            except ValueError:
                createVar['origin_p{}'.format(i)] = wfdb.rdrecord('./data/' + file_num, channels=[1])
                createVar['origin_p{}'.format(i)] = np.squeeze(
                    createVar['origin_p{}'.format(i)].p_signal[self.washout:self.washout + self.length])

    def _peakfinding(self):
        for i in range(self.num_series):
            createVar['detrend_p{}'.format(i)] = nk.signal_detrend(np.squeeze(createVar['origin_p{}'.format(i)]),order=5)
            createVar['clean_p{}'.format(i)] = nk.ppg_clean(createVar['detrend_p{}'.format(i)], sampling_rate=1000)
            createVar['peak_p_new{}'.format(i)] = utility.ppg_findpeaks_elgendi(createVar['clean_p{}'.format(i)],
                                                                        sampling_rate=1000, peakwindow=self.peakwindow,
                                                                        beatwindow=self.beatwindow)[1:]

    def _segment_decision(self):
        self.length_all = []
        for i in range(0, self.num_series):
            createVar['seg_p{}'.format(i)] = []
            for x, y in zip(createVar['peak_p_new{}'.format(i)][0:-1], createVar['peak_p_new{}'.format(i)][1:]):
                createVar['seg_p{}'.format(i)].append(createVar['origin_p{}'.format(i)][x:y])
            for l in range(len(createVar['seg_p{}'.format(i)])):
                self.length_all.append(createVar['seg_p{}'.format(i)][l].size)
                if l == 0:
                    createVar['abs_large_value_p{}'.format(i)] = np.abs(np.max(createVar['seg_p{}'.format(i)][l])) \
                        if np.abs(np.max(createVar['seg_p{}'.format(i)][l])) > np.abs(
                        np.min(createVar['seg_p{}'.format(i)][l])) \
                        else np.abs(np.min(createVar['seg_p{}'.format(i)][l]))
                else:
                    createVar['abs_large_value_p{}'.format(i)] = np.max(np.array(
                        [np.abs(np.max(createVar['seg_p{}'.format(i)][l])),
                         np.abs(np.min(createVar['seg_p{}'.format(i)][l]))])) \
                        if np.max(np.array([np.abs(np.max(createVar['seg_p{}'.format(i)][l])),
                                            np.abs(np.min(createVar['seg_p{}'.format(i)][l]))])) > createVar[
                               'abs_large_value_p{}'.format(i)] \
                        else createVar['abs_large_value_p{}'.format(i)]

    def _preprocessing(self):
        self.max_length = max(self.length_all)
        self.min_length = min(self.length_all)
        self.mean_length = int(statistics.mean(self.length_all))
        self.median = int(statistics.median(self.length_all))
        self.median_low = int(statistics.median_low(self.length_all))
        self.median_high = int(statistics.median_high(self.length_all))
        for i in range(0, self.num_series):
            if i == 0:
                self.abs_large_value = createVar['abs_large_value_p{}'.format(i)]
            else:
                self.abs_large_value = createVar['abs_large_value_p{}'.format(i)] if createVar[
                                                                                         'abs_large_value_p{}'.format(
                                                                                             i)] > self.abs_large_value \
                    else self.abs_large_value

    def _flipping(self):
        for i in range(0, self.num_series):
            for j in range(len(createVar['seg_p{}'.format(i)])):
                createVar['seg_p{}'.format(i)][j] = np.flipud(createVar['seg_p{}'.format(i)][j])

    def _padding(self):
        for i in range(0, self.num_series):
            for j in range(len(createVar['seg_p{}'.format(i)])):
                createVar['seg_p{}'.format(i)][j] = np.concatenate((np.zeros(
                    self.max_length - createVar['seg_p{}'.format(i)][j].size), createVar['seg_p{}'.format(i)][j]),
                                                                   0)

    def _normalization(self):
        for i in range(0, self.num_series):
            for j in range(len(createVar['seg_p{}'.format(i)])):
                createVar['seg_p{}'.format(i)][j] /= self.abs_large_value

    def _dataset_partition(self):
        train_end_pos = int(self.training_ratio * self.num_select_seg)
        if self.validation == True:
            validation_end_pos = int(train_end_pos + self.validation_ratio * self.num_select_seg)
            test_end_pos = int(validation_end_pos + self.testing_ratio * self.num_select_seg)
        else:
            validation_end_pos = None
            test_end_pos = int(train_end_pos + self.testing_ratio * self.num_select_seg)
        for i in range(0, self.num_series):
            if i == 0:
                self.train_x = []
                self.validation_x = []
                self.test_x = []
                self.train_y = []
                self.validation_y = []
                self.test_y = []
            self.train_x.append(createVar['seg_p{}'.format(i)][:train_end_pos])
            self.train_y.append(np.repeat(i, int(self.training_ratio * self.num_select_seg)))
            if validation_end_pos is not None:
                self.validation_x.append(createVar['seg_p{}'.format(i)][train_end_pos:validation_end_pos])
                self.validation_y.append(np.repeat(i, int(self.validation_ratio * self.num_select_seg)))
                self.test_x.append(createVar['seg_p{}'.format(i)][validation_end_pos:test_end_pos])
                self.test_y.append(np.repeat(i, int(self.validation_ratio * self.num_select_seg)))
            else:
                 self.test_x.append(createVar['seg_p{}'.format(i)][train_end_pos:test_end_pos])
                 self.test_y.append(np.repeat(i,int(self.testing_ratio*self.num_select_seg)))

        #        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        #        self.validation_x = np.array(self.validation_x)
        self.validation_y = np.array(self.validation_y)
        #        self.test_x = np.array(self.test_x)
        self.test_y = np.array(self.test_y)

    def from_3d_to_2d(self):
        #self.train_x = self.train_x.reshape(self.train_x.shape[0]*self.train_x.shape[1],self.train_x.shape[2])
        self.train_y = self.train_y.reshape(self.train_y.shape[0] * self.train_y.shape[1])
        if self.validation is True:
            #self.validation_x = self.validation_x.reshape(self.validation_x.shape[0]*self.validation_x.shape[1],self.validation_x.shape[2])
            self.validation_y = self.validation_y.reshape(self.validation_y.shape[0] * self.validation_y.shape[1])
        #self.test_x = self.test_x.reshape(self.test_x.shape[0]*self.test_x.shape[1],self.test_x.shape[2])
        self.test_y = self.test_y.reshape(self.test_y.shape[0] * self.test_y.shape[1])

if __name__ == "__main__":
    dataset = timeseries_loader()
    dataset
