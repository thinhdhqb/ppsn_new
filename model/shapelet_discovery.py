import numpy as np
import util.auto_pisd as auto_pisd
import util.pst_support_method as pstsm
import util.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial
import random

class ShapeletDiscover():
    def __init__(self, window_size=20, num_pip=0.4, processes=4):
        self.window_size = window_size
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = None
        self.list_labels = None
        self.processes = processes

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_shapelet_info(self, number_of_shapelet):
        number_of_shapelet = int(number_of_shapelet * self.len_of_ts)
        number_of_shapelet = round(number_of_shapelet / len(self.list_labels))
        if number_of_shapelet == 0:
            number_of_shapelet = 1

        # Find number_of_shapelet from each group and sort by start position
        list_group_shapelet = pstsm.find_c_shapelet(self.list_group_ppi[0], number_of_shapelet) 
        list_group_shapelet = np.asarray(list_group_shapelet)
        list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
        list_shapelet = list_group_shapelet
        for i in range(1, len(self.list_group_ppi)):
            list_ppi = self.list_group_ppi[i]
            list_group_shapelet = pstsm.find_c_shapelet(list_ppi, number_of_shapelet)
            list_group_shapelet = np.asarray(list_group_shapelet)
            list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
            list_shapelet = np.concatenate((list_shapelet,list_group_shapelet),axis=0)

        return list_shapelet

    def find_ppi(self, i, l):
        # Find PPI for time series i in group l
        list_result = []
        ts_pos = self.group_train_data_pos[l][i] # Original time series position
        pdm = {} # Pairwise distance matrix
        t1 = self.group_train_data[l][i] # Current time series
        # print(l,ts_pos,i)
        pdm[i * 10000 + i] = np.zeros((0, 0))
        for p in range(len(self.train_data)):
            t2 = self.train_data[p]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
            pdm[ts_pos * 10000 + p] = matrix_1

        # Calculate information gain for each shapelet candidate in time series i
        for j in range(len(self.group_train_data_piss[l][i])):
            ts_pis = self.group_train_data_piss[l][i][j] # Current shapelet candidate
            ts_ci_pis = self.group_train_data_ci_piss[l][i][j] # Current shapelet candidate with complexity invariant
            # Calculate subdist with all time series
            list_dist = []
            for p in range(len(self.train_data)):
                if p == ts_pos:
                    list_dist.append(0)
                else:
                    matrix = pdm[ts_pos * 10000 + p]
                    ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)
                    ts_2_ci = self.train_data_ci[p]
                    pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                    dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos,
                                                   self.list_end_pos, ts_ci_pis, pcs_ci_list)
                    list_dist.append(dist)

            # Calculate best information gain
            ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
            # PPI format: [time series position, start position, end position, information gain, label]
            ppi = np.asarray([ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l]]) 
            list_result.append(ppi)
        return list_result

    def extract_candidate(self, train_data):
        # Extract shapelet candidate
        time1 = time.time()
        self.train_data_piss = np.asarray(
            [auto_pisd.auto_piss_extractor(train_data[i], num_pip=self.num_pip) for i in range(len(train_data))], dtype="object")
        ci_return = [auto_pisd.auto_ci_extractor(train_data[i], self.train_data_piss[i]) for i in range(len(train_data))]
        self.train_data_ci = np.asarray([ci_return[i][0] for i in range(len(ci_return))], dtype="object")
        self.train_data_ci_piss = np.asarray([ci_return[i][1] for i in range(len(ci_return))], dtype="object")
        time1 = time.time() - time1
        print("extracting time: %s" % time1)


    def discovery(self, train_data, train_labels):
        self.window_size = int(self.window_size)
        time2 = time.time()
        self.train_data = train_data
        self.train_labels = train_labels

        self.len_of_ts = len(train_data[0])
        self.list_labels = np.unique(train_labels) # Get unique labels from training data

        self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
        self.list_end_pos = np.ones(self.len_of_ts, dtype=int) * (self.window_size * 2 + 1)
        for i in range(self.window_size):
            self.list_end_pos[-(i + 1)] -= self.window_size - i
        for i in range(self.window_size - 1):
            self.list_start_pos[i] += self.window_size - i - 1
        num_candidate = sum(len(i) for i in self.train_data_ci)
        print(f"num candidate: {num_candidate}")
        # Divide time series into group of label
        group_train_data = [[] for i in self.list_labels]
        group_train_data_pos = [[] for i in self.list_labels]
        group_train_data_piss = [[] for i in self.list_labels]
        group_train_data_ci_piss = [[] for i in self.list_labels]

        for l in range(len(self.list_labels)): # For each label (l is index of label)
            for i in range(len(train_data)): # For each time series
                # If the label of time series is equal to the label of group
                if train_labels[i] == self.list_labels[l]:
                    group_train_data[l].append(train_data[i]) # append time series to group
                    group_train_data_pos[l].append(i)
                    group_train_data_piss[l].append(self.train_data_piss[i])
                    group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])

        self.group_train_data = np.asarray(group_train_data, dtype="object")
        self.group_train_data_pos = np.asarray(group_train_data_pos, dtype="object")
        self.group_train_data_piss = np.asarray(group_train_data_piss, dtype="object")
        self.group_train_data_ci_piss = np.asarray(group_train_data_ci_piss, dtype="object")

        # Select shapelet
        self.list_group_ppi = []
        for l in range(len(self.list_labels)):
            p = multiprocessing.Pool(processes=self.processes)
            temp_ppi = p.map(partial(self.find_ppi, l=l), range(len(self.group_train_data[l]))) # call find_ppi for each time series in group l
            
            list_ppi = []
            # For each time series in group l, append ppi to list_ppi
            for i in range(len(self.group_train_data[l])):
                pii_in_i = temp_ppi[i]
                for j in range(len(pii_in_i)):
                    list_ppi.append(pii_in_i[j])
            list_ppi = np.asarray(list_ppi)
            self.list_group_ppi.append(list_ppi)
        time2 = time.time() - time2
        print("window_size: %s - evaluating_time: %s" % (self.window_size, time2))
        return time2,num_candidate
