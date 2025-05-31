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
    def __init__(self, window_sizes = [10, 20, 30, 50, 100, 200], num_pip=0.4, processes=4, subset_ratio=0.2, r=3):
        self.default_window_sizes = window_sizes
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = None
        self.list_labels = None
        self.processes = processes
        self.subset_ratio = subset_ratio  # Ratio of training data used in first round (default 20%)
        self.r = r  # Value of r (default 3)

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

    def find_ppi(self, i, l, subset_indices=None):
        # Find PPI for time series i in group l
        list_result = []
        ts_pos = self.group_train_data_pos[l][i] # Original time series position
        pdm = {} # Pairwise distance matrix
        t1 = self.group_train_data[l][i]
        # pdm[i * 10000 + i] = np.zeros((0, 0))
        
        # If subset_indices is provided, only calculate distances to time series in the subset
        if subset_indices is not None:
            eval_indices = subset_indices
        else:
            eval_indices = range(len(self.train_data))
            
        for window_size in self.window_sizes:
            for p in eval_indices:
                t2 = self.train_data[p]
                matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, window_size)
                pdm[f"{ts_pos}_{p}_{window_size}"] = matrix_1

        # Calculate information gain for each shapelet candidate in time series i
        for j in range(len(self.group_train_data_piss[l][i])):
            ts_pis = self.group_train_data_piss[l][i][j] # Current shapelet candidate
            ts_ci_pis = self.group_train_data_ci_piss[l][i][j] # Current shapelet candidate with complexity invariant

            best_ig = -1
            best_window_size = None
            best_dist_list = None
            # Calculate subdist with time series in evaluation set for each window size
            for window_size in self.window_sizes:
                list_dist = [0] * len(self.train_data)  # Initialize with zeros for all time series
                print("List start pos for window size %s: %s" % (window_size, self.list_start_pos[window_size]))
                print("List end pos for window size %s: %s" % (window_size, self.list_end_pos[window_size]))
                for p in eval_indices:
                    if p == ts_pos:
                        list_dist[p] = 0
                    else:
                        matrix = pdm[f"{ts_pos}_{p}_{window_size}"]
                        ts_pcs = auto_pisd.pcs_extractor(ts_pis, window_size, self.len_of_ts)
                        ts_2_ci = self.train_data_ci[p]
                        pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                        dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos[window_size],
                                                    self.list_end_pos[window_size], ts_ci_pis, pcs_ci_list)
                        print("Distance for time series %s with shapelet %s: %s" % (p, ts_pis, dist))
                        list_dist[p] = dist
                # Calculate information gain for current window size
                if subset_indices is not None:
                    subset_dist = [list_dist[idx] for idx in subset_indices]
                    subset_labels = [self.train_labels[idx] for idx in subset_indices]
                    current_ig = ssm.find_best_split_point_and_info_gain(subset_dist, subset_labels, self.list_labels[l])
                else:
                    current_ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
                # print("Current IG: %s for window size %s" % (current_ig, window_size))
                # Update best if current is better
                if current_ig > best_ig:
                    # print("Found better IG: %s for window size %s" % (current_ig, window_size))
                    best_ig = current_ig
                    # print("Best IG: %s" % best_ig)
                    best_window_size = window_size
                    # print("Best window size: %s" % best_window_size)

            idx_c_on_ts = i
            ppi = np.asarray([ts_pos, ts_pis[0], ts_pis[1], best_ig, self.list_labels[l],ts_ci_pis,idx_c_on_ts, best_window_size])
            list_result.append(ppi)
            break
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
        Two-phase shapelet discovery process
        Parameters:
        train_data: list of time series data
        train_labels: list of corresponding labels
        g: number of final shapelets to select
        r: expansion factor for round 1 selection (default to self.r if not provided)
        """
        print("len data"+str(len(train_data)))
        time_start = time.time()
        self.train_data = train_data
        self.train_labels = train_labels
        self.len_of_ts = len(train_data[0])

        self.window_sizes = [window_size for window_size in self.default_window_sizes if window_size < self.len_of_ts]  # Default window sizes
        self.list_labels = np.unique(train_labels)

        self.list_start_pos = {}
        self.list_end_pos = {}

        for window_size in self.window_sizes:
            self.list_start_pos[window_size] = np.ones(self.len_of_ts, dtype=int)
            self.list_end_pos[window_size] = np.ones(self.len_of_ts, dtype=int) * (window_size * 2 + 1)
            for i in range(window_size):
                self.list_end_pos[window_size][-(i + 1)] -= window_size - i
            for i in range(window_size - 1):
                self.list_start_pos[window_size][i] += window_size - i - 1
        num_candidate = sum(len(i) for i in self.train_data_ci)
        print(f"num candidate: {num_candidate}")

        # Divide time series into group of label
        self._group_data_by_label(train_data)

        # PHASE 1: Select candidates on a subset of the data
        print("Phase 1: Initial candidate evaluation on subset")
        subset_size = int(len(train_data) * self.subset_ratio)
        subset_indices = random.sample(range(len(train_data)), subset_size)

        # First round using subset of data

        round1_group_ppi = []
        for l in range(len(self.list_labels)):
            p = multiprocessing.Pool(processes=self.processes)
            temp_ppi = p.map(partial(self.find_ppi, l=l, subset_indices=subset_indices), range(len(self.group_train_data[l])))
            p.close()
            p.join()
            list_ppi = []
            for i in range(len(self.group_train_data[l])):
                pii_in_i = temp_ppi[i]
                for j in range(len(pii_in_i)):
                    list_ppi.append(pii_in_i[j])
            list_ppi = np.asarray(list_ppi)
            round1_group_ppi.append(list_ppi)
            del temp_ppi
            del list_ppi
            gc.collect()  
        number_of_shapelet_per_class = -1

        # Calculate number of candidates to select per class for round 1
        number_of_shapelet_per_class = int(g * self.len_of_ts)
        number_of_shapelet_per_class = round(number_of_shapelet_per_class / len(self.list_labels))
        if number_of_shapelet_per_class == 0:
            number_of_shapelet_per_class = 1
        # number_of_shapelet_per_class = int(g)

        # Select top g*r candidates from round 1
        print("Sorting")
        self.candidates = [[] for _ in self.list_labels]
        num_candidate_select = 0
        for l, label in enumerate(self.list_labels):
            sorted_candidates = round1_group_ppi[l][round1_group_ppi[l][:, 3].argsort()[::-1]]
            top_candidates = sorted_candidates[:number_of_shapelet_per_class * self.r]
            self.candidates[l].extend(top_candidates)
            num_candidate_select +=len(top_candidates)  

        print(f"Phase 2: Re-evaluating {   num_candidate_select} candidates on full dataset")

        print("evaluating")
        timee = time.time()
        for l in range(len(self.list_labels)):
            with multiprocessing.Pool(processes=self.processes) as p:
                n_tasks = len(self.candidates[l])  
                n_procs = self.processes               
                factor = 4
                chunksize = max(1, n_tasks // (n_procs * factor)) 
                temp_ppi = p.map(partial(self.evaluate_group_wrapper_new), self.candidates[l],chunksize)
            list_ppi = [ppi for sublist in temp_ppi for ppi in sublist]  # flatten
            list_ppi = np.asarray(list_ppi)

            self.list_group_ppi.append(list_ppi)
            del temp_ppi
            del list_ppi
            gc.collect()  

        print(time.time()-timee)
        time_total = time.time() - time_start
        print(f"Total time: {time_total}")
        return time_total,num_candidate,num_candidate_select

    def evaluate_group_wrapper_new(self,c):
        results = []
        num_train_samples = len(self.train_data)

        # group_data_pos = np.array(self.group_train_data_pos[label])
        # group_data = self.group_train_data[label]
        # for c in candidates[label]:
        label = int(c[4])
        ts_pos = int(c[0])
        start_pos = int(c[1])
        end_pos = int(c[2])
        ts_ci_pis = c[5]
        ts_pis = [start_pos, end_pos]
        ts_idx = int(c[6])
        window_size = int(c[7])
        pdm = {}
        t1 = self.group_train_data[label][ts_idx]
        pdm[ts_idx * 10000 + ts_idx] = np.zeros((0, 0))

        for p in range(num_train_samples):
            t2 = self.train_data[p]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, window_size)
            pdm[ts_pos * 10000 + p] = matrix_1
        # Chuẩn bị đoạn pcs một lần
        ts_pcs = auto_pisd.pcs_extractor(ts_pis, window_size, self.len_of_ts)

        list_dist = np.zeros(len(self.train_data))

        for p in range(num_train_samples):
            if p == ts_pos:
                continue  # Khoảng cách với chính nó là 0

            matrix = pdm[ts_pos * 10000 + p]
            ts_2_ci = self.train_data_ci[p]
            pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
            dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos[window_size],
                                            self.list_end_pos[window_size], ts_ci_pis, pcs_ci_list)
            # Tính khoảng cách
            list_dist[p] = dist
        # Tính IG
        ig = ssm.find_best_split_point_and_info_gain(
            list_dist, self.train_labels, self.list_labels[label]
        )
        results.append(np.array([ts_pos, start_pos, end_pos, ig, label, window_size]))

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