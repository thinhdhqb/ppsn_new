import numpy as np
import util.auto_pisd as auto_pisd
import util.pst_support_method as pstsm
import util.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial
import random

class ShapeletDiscover():
    def __init__(self, window_size=20, num_pip=0.4, processes=4, r=3, subset_ratio=0.2):
        self.window_size = window_size
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = None
        self.list_labels = None
        self.processes = processes
        self.r = r  # Tỉ lệ mở rộng cho số lượng candidate ở vòng 1
        self.subset_ratio = subset_ratio  # Tỉ lệ của tập con cho vòng 1 (20%)

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_shapelet_info(self, number_of_shapelet):
        number_of_shapelet = int(number_of_shapelet * self.len_of_ts)
        number_of_shapelet = round(number_of_shapelet / len(self.list_labels))
        if number_of_shapelet == 0:
            number_of_shapelet = 1

        # Phase 1: Đánh giá các candidate trên tập con
        subset_candidates = self._evaluate_candidates_on_subset(number_of_shapelet * self.r)
        
        # Phase 2: Đánh giá lại các candidates đã chọn trên toàn bộ tập dữ liệu
        final_shapelets = self._evaluate_candidates_on_full_dataset(subset_candidates, number_of_shapelet)
        
        return final_shapelets

    def _evaluate_candidates_on_subset(self, number_of_candidates):
        """
        Vòng 1: Đánh giá candidates trên một tập con (20%) của tập huấn luyện
        và chọn ra top g*r candidates.
        """
        print("Phase 1: Evaluating candidates on subset...")
        # Tạo một tập con ngẫu nhiên từ tập dữ liệu huấn luyện
        subset_size = int(len(self.train_data) * self.subset_ratio)
        subset_indices = random.sample(range(len(self.train_data)), subset_size)
        
        subset_train_data = [self.train_data[i] for i in subset_indices]
        subset_train_labels = [self.train_labels[i] for i in subset_indices]
        
        # Lưu lại tập dữ liệu gốc
        original_train_data = self.train_data
        original_train_labels = self.train_labels
        
        # Tạm thời thay thế bằng tập con để đánh giá
        self.train_data = subset_train_data
        self.train_labels = subset_train_labels
        
        # Thực hiện tìm kiếm PPI trên tập con
        subset_list_group_ppi = []
        for l in range(len(self.list_labels)):
            p = multiprocessing.Pool(processes=self.processes)
            temp_ppi = p.map(partial(self.find_ppi, l=l), range(len(self.group_train_data[l])))
            p.close()
            p.join()

            list_ppi = []
            for i in range(len(self.group_train_data[l])):
                pii_in_i = temp_ppi[i]
                for j in range(len(pii_in_i)):
                    list_ppi.append(pii_in_i[j])
            list_ppi = np.asarray(list_ppi)
            subset_list_group_ppi.append(list_ppi)
        
        # Khôi phục lại tập dữ liệu gốc
        self.train_data = original_train_data
        self.train_labels = original_train_labels
        
        # Tìm top candidates từ mỗi nhóm label
        all_candidates = []
        for l in range(len(self.list_labels)):
            list_ppi = subset_list_group_ppi[l]
            candidates_per_label = int(number_of_candidates / len(self.list_labels))
            list_group_shapelet = pstsm.find_c_shapelet(list_ppi, candidates_per_label)
            list_group_shapelet = np.asarray(list_group_shapelet)
            if len(list_group_shapelet) > 0:
                list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
                all_candidates.extend(list_group_shapelet)
        
        print(f"Phase 1 completed: Selected {len(all_candidates)} candidates")
        return np.array(all_candidates)

    def _evaluate_candidates_on_full_dataset(self, candidates, number_of_shapelet):
        """
        Vòng 2: Đánh giá lại các candidates đã chọn ở Vòng 1 trên toàn bộ tập huấn luyện
        và chọn ra top g candidates cuối cùng.
        """
        print("Phase 2: Re-evaluating candidates on full dataset...")
        
        # Chuyển đổi shapelet candidates thành định dạng ppi để đánh giá lại
        reevaluated_candidates = []
        
        for candidate in candidates:
            ts_pos = int(candidate[0])
            start_pos = int(candidate[2])
            end_pos = int(candidate[3])
            label = int(candidate[4])
            
            # Tìm time series từ ts_pos
            ts = self.train_data[ts_pos]
            pis = [start_pos, end_pos]
            
            # Tính CI cho shapelet này
            ts_ci_extracted = auto_pisd.auto_ci_extractor(ts, [np.array(pis)])
            ts_ci = ts_ci_extracted[1][0]
            
            # Tính toán khoảng cách và information gain trên toàn bộ tập dữ liệu
            list_dist = []
            pdm = {}
            
            # Tính toán ma trận khoảng cách cho shapelet candidate
            for p in range(len(self.train_data)):
                if p == ts_pos:
                    list_dist.append(0)
                else:
                    t2 = self.train_data[p]
                    # Tính toán ma trận khoảng cách giữa ts và t2
                    if ts_pos * 10000 + p not in pdm:
                        matrix_1, _ = auto_pisd.calculate_matrix(ts, t2, self.window_size)
                        pdm[ts_pos * 10000 + p] = matrix_1
                    else:
                        matrix_1 = pdm[ts_pos * 10000 + p]
                    
                    ts_pcs = auto_pisd.pcs_extractor(pis, self.window_size, self.len_of_ts)
                    
                    # Tính CI list cho time series thứ 2
                    ts_2_ci = auto_pisd.auto_ci_extractor(t2, [])[0]
                    if ts_pcs[0] >= len(ts_2_ci) or ts_pcs[0] < 0:
                        list_dist.append(float("inf"))
                        continue
                        
                    end_idx = min(ts_pcs[1] - 1, len(ts_2_ci))
                    if end_idx <= ts_pcs[0]:
                        list_dist.append(float("inf"))
                        continue
                        
                    pcs_ci_list = ts_2_ci[ts_pcs[0]:end_idx]
                    
                    # Sử dụng hàm find_min_dist đã định nghĩa ở đầu file
                    dist = auto_pisd.find_min_dist(pis, ts_pcs, matrix_1,
                                         self.list_start_pos, self.list_end_pos, ts_ci, pcs_ci_list)
                    list_dist.append(dist)
            
            # Tính information gain mới trên toàn bộ tập dữ liệu
            ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, label)
            updated_candidate = np.asarray([ts_pos, ig, start_pos, end_pos, label])
            reevaluated_candidates.append(updated_candidate)
        
        # Sắp xếp candidate theo information gain và chọn top number_of_shapelet
        reevaluated_candidates = np.array(reevaluated_candidates)
        if len(reevaluated_candidates) > 0:
            reevaluated_candidates = reevaluated_candidates[reevaluated_candidates[:, 1].argsort()][::-1]
            final_candidates = reevaluated_candidates[:number_of_shapelet * len(self.list_labels)]
        else:
            final_candidates = reevaluated_candidates
        
        print(f"Phase 2 completed: Selected final {len(final_candidates)} shapelets")
        return final_candidates

    def find_ppi(self, i, l):
        list_result = []
        ts_pos = self.group_train_data_pos[l][i]
        pdm = {}
        t1 = self.group_train_data[l][i]
        pdm[i * 10000 + i] = np.zeros((0, 0))
        for p in range(len(self.train_data)):
            t2 = self.train_data[p]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
            pdm[ts_pos * 10000 + p] = matrix_1

        for j in range(len(self.group_train_data_piss[l][i])):
            ts_pis = self.group_train_data_piss[l][i][j]
            ts_ci_pis = self.group_train_data_ci_piss[l][i][j]
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
        self.list_labels = np.unique(train_labels)

        self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
        self.list_end_pos = np.ones(self.len_of_ts, dtype=int) * (self.window_size * 2 + 1)
        for i in range(self.window_size):
            self.list_end_pos[-(i + 1)] -= self.window_size - i
        for i in range(self.window_size - 1):
            self.list_start_pos[i] += self.window_size - i - 1

        # Divide time series into group of label
        group_train_data = [[] for i in self.list_labels]
        group_train_data_pos = [[] for i in self.list_labels]
        group_train_data_piss = [[] for i in self.list_labels]
        group_train_data_ci_piss = [[] for i in self.list_labels]

        for l in range(len(self.list_labels)):
            for i in range(len(train_data)):
                if train_labels[i] == self.list_labels[l]:
                    group_train_data[l].append(train_data[i])
                    group_train_data_pos[l].append(i)
                    group_train_data_piss[l].append(self.train_data_piss[i])
                    group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])

        self.group_train_data = np.asarray(group_train_data, dtype="object")
        self.group_train_data_pos = np.asarray(group_train_data_pos, dtype="object")
        self.group_train_data_piss = np.asarray(group_train_data_piss, dtype="object")
        self.group_train_data_ci_piss = np.asarray(group_train_data_ci_piss, dtype="object")

       
        time2 = time.time() - time2
        print("Discovery preparation time: %s" % time2)