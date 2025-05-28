import gc
import numpy as np
import util.auto_pisd as auto_pisd
import util.pst_support_method as pstsm
import util.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial
import random

class ShapeletDiscover():
    def __init__(self, window_sizes=[5, 10, 20, 30, 50, 100, 200], num_pip=0.4, processes=4, subset_ratio=0.2, r=3):
        self.window_sizes = window_sizes
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = None
        self.list_labels = None
        self.processes = processes
        self.subset_ratio = subset_ratio
        self.r = r

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_shapelet_info(self, number_of_shapelet):
        # if number_of_shapelet < 1:
        number_of_shapelet = int(number_of_shapelet * self.len_of_ts)
        number_of_shapelet = round(number_of_shapelet / len(self.list_labels))
        if number_of_shapelet == 0:
            number_of_shapelet = 1  
        # number_of_shapelet = int(number_of_shapelet)
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
    # def get_shapelet_after_round1(self,num_of_shaplet):
    #     number_of_shapelet_per_class = int(num_of_shaplet * self.len_of_ts)
    #     number_of_shapelet_per_class = round(number_of_shapelet_per_class / len(self.list_labels))
    #     if number_of_shapelet_per_class == 0:
    #         number_of_shapelet_per_class = 1
        
    #     return
    def find_ppi(self, i, l, subset_indices=None):

        list_result = []
        ts_pos = self.group_train_data_pos[l][i]
        # print(ts_pos)
        pdm = {}
        t1 = self.group_train_data[l][i]
        # print(t1)
        pdm[i * 10000 + i] = np.zeros((0, 0))
        
        # If subset_indices is provided, only calculate distances to time series in the subset
        if subset_indices is not None:
            eval_indices = subset_indices
        else:
            eval_indices = range(len(self.train_data))
            
        for p in eval_indices:
            t2 = self.train_data[p]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
            pdm[ts_pos * 10000 + p] = matrix_1

        for j in range(len(self.group_train_data_piss[l][i])):
            ts_pis = self.group_train_data_piss[l][i][j]
            print(ts_pis)
            ts_ci_pis = self.group_train_data_ci_piss[l][i][j]
            # print(ts_ci_pis)

            # Calculate subdist with time series in evaluation set
            list_dist = [0] * len(self.train_data)  # Initialize with zeros for all time series
            
            for p in eval_indices:
                if p == ts_pos:
                    list_dist[p] = 0
                else:
                    matrix = pdm[ts_pos * 10000 + p]
                    ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)
                    ts_2_ci = self.train_data_ci[p]
                    pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                    dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos,
                                                   self.list_end_pos, ts_ci_pis, pcs_ci_list)
                    list_dist[p] = dist

            # If using subset, only consider the subset labels for information gain
            if subset_indices is not None:
                subset_dist = [list_dist[idx] for idx in subset_indices]
                subset_labels = [self.train_labels[idx] for idx in subset_indices]
                ig = ssm.find_best_split_point_and_info_gain(subset_dist, subset_labels, self.list_labels[l])
            else:
                ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
            idx_c_on_ts = i
            ppi = np.asarray([ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l],ts_ci_pis,idx_c_on_ts])
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
    
    
    def discovery(self, train_data, train_labels, g):
        """
        Multi-window shapelet discovery process
        Parameters:
        train_data: list of time series data
        train_labels: list of corresponding labels
        g: number of final shapelets to select
        """
        print("len data"+str(len(train_data)))
        time_start = time.time()
        self.train_data = train_data
        self.train_labels = train_labels
        self.len_of_ts = len(train_data[0])
        self.list_labels = np.unique(train_labels)

        all_candidates = []
        total_candidates = 0
        
        # Process each window size
        for window_size in self.window_sizes:
            print(f"Processing window size: {window_size}")
            self.window_size = int(window_size)
            
            # Setup position lists for current window size
            self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
            self.list_end_pos = np.ones(self.len_of_ts, dtype=int) * (self.window_size * 2 + 1)
            for i in range(self.window_size):
                self.list_end_pos[-(i + 1)] -= self.window_size - i
            for i in range(self.window_size - 1):
                self.list_start_pos[i] += self.window_size - i - 1

            # Extract candidates for current window size
            self.extract_candidate(train_data)
            num_candidate = sum(len(i) for i in self.train_data_ci)
            total_candidates += num_candidate
            print(f"Number of candidates for window size {window_size}: {num_candidate}")

            # Group data by label
            self._group_data_by_label(train_data)
            
            # Phase 1: Evaluate on subset
            subset_size = int(len(train_data) * self.subset_ratio)
            subset_indices = random.sample(range(len(train_data)), subset_size)
            window_candidates = self._evaluate_on_subset(subset_indices)
            
            # Add window size information to candidates
            for label_candidates in window_candidates:
                for candidate in label_candidates:
                    candidate = np.append(candidate, window_size)  # Add window size as last element
                    all_candidates.append(candidate)

        # Phase 2: Select top g candidates across all window sizes
        print(f"Total candidates across all window sizes: {len(all_candidates)}")
        all_candidates = np.array(all_candidates)
        sorted_candidates = all_candidates[all_candidates[:, 3].argsort()[::-1]]  # Sort by information gain
        top_candidates = sorted_candidates[:g]
        
        # Store final candidates
        self.list_group_ppi = []
        for label in self.list_labels:
            label_candidates = top_candidates[top_candidates[:, 4] == label]
            self.list_group_ppi.append(label_candidates)

        time_total = time.time() - time_start
        print(f"Total discovery time: {time_total}")
        return time_total, total_candidates, g

    def evaluate_group_wrapper_new(self, c):
        results = []
        num_train_samples = len(self.train_data)
        
        label = int(c[4])
        ts_pos = int(c[0])
        start_pos = int(c[1])
        end_pos = int(c[2])
        ts_ci_pis = c[5]
        window_size = int(c[-1])  # Get window size from candidate
        ts_pis = [start_pos, end_pos]
        ts_idx = int(c[6])
        
        # Use the correct window size for this candidate
        self.window_size = window_size
        
        pdm = {}
        t1 = self.group_train_data[label][ts_idx]
        pdm[ts_idx * 10000 + ts_idx] = np.zeros((0, 0))

        for p in range(num_train_samples):
            t2 = self.train_data[p]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
            pdm[ts_pos * 10000 + p] = matrix_1
        # Chuẩn bị đoạn pcs một lần
        ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)

        list_dist = np.zeros(len(self.train_data))

        for p in range(num_train_samples):
            if p == ts_pos:
                continue  # Khoảng cách với chính nó là 0

            matrix = pdm[ts_pos * 10000 + p]
            ts_2_ci = self.train_data_ci[p]
            pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
            dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos,
                                            self.list_end_pos, ts_ci_pis, pcs_ci_list)
            # Tính khoảng cách
            list_dist[p] = dist
        # Tính IG
        ig = ssm.find_best_split_point_and_info_gain(
            list_dist, self.train_labels, self.list_labels[label]
        )
        results.append(np.array([ts_pos, start_pos, end_pos, ig, label]))

        return results

    def _group_data_by_label(self, train_data):
        group_train_data = [[] for _ in self.list_labels]
        group_train_data_pos = [[] for _ in self.list_labels]
        group_train_data_piss = [[] for _ in self.list_labels]
        group_train_data_ci_piss = [[] for _ in self.list_labels]

        for l in range(len(self.list_labels)):
            for i in range(len(train_data)):
                if self.train_labels[i] == self.list_labels[l]:
                    group_train_data[l].append(train_data[i])
                    group_train_data_pos[l].append(i)
                    group_train_data_piss[l].append(self.train_data_piss[i])
                    group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])

        self.group_train_data = np.asarray(group_train_data, dtype="object")
        self.group_train_data_pos = np.asarray(group_train_data_pos, dtype="object")
        self.group_train_data_piss = np.asarray(group_train_data_piss, dtype="object")
        self.group_train_data_ci_piss = np.asarray(group_train_data_ci_piss, dtype="object")
