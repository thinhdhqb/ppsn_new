import csv
import subprocess
import os

# Đọc danh sách dataset đã xử lý
def read_processed_datasets(log_file="processed_datasets.txt"):
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r") as f:
        return set(line.strip() for line in f if line.strip())

# Ghi thêm một dataset đã xử lý vào log
def append_processed_dataset(dataset, log_file="processed_datasets.txt"):
    with open(log_file, "a") as f:
        f.write(f"{dataset}\n")

def read_folder_sizes(file_path):
    sizes = {}
    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                name, size_str = line.strip().split(":")
                size = float(size_str.replace("MB", "").strip())
                sizes[name.strip()] = size
    return sizes

def read_config_csv(file_path):
    configs = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = row['Dataset']
            num_shapelet = row['number_of_shapelets']
            window_size = row['window_size']
            configs[dataset] = (num_shapelet, window_size)
    return configs

def run_filtered_commands(folder_file, config_file, max_datasets=10):
    folder_sizes = read_folder_sizes(folder_file)
    configs = read_config_csv(config_file)
    success_log = []

    processed_datasets = read_processed_datasets()
    count = 0

    for dataset, size in folder_sizes.items():
        # if dataset in processed_datasets:
        #     print(f"✅ Đã xử lý '{dataset}', bỏ qua.")
        #     continue

        if dataset in configs:
            num_shapelet, window_size = configs[dataset]
            # if size < 2 and float(num_shapelet) >= 1:
            if size < 2 :
                output_file_new = f"train/{dataset}_new.txt"
                output_file_old = f"train/{dataset}.txt"
                result_file_old = f"{dataset}_result.txt"
                result_file_new = f"{dataset}_result_new.txt"

                cmd_new = (
                    f'python3 ppsn_demo_newv3.py '
                    f'--num_shapelet={str(num_shapelet)} '
                    f'--window_size={window_size} --epochs=200 | tee {output_file_new}'
                    
                )

                cmd_old = (
                    f'python3 ppsn_demo.py '
                    f'--dataset_name="{dataset}" '
                    f'--num_shapelet={num_shapelet} '
                    f'--window_size={window_size} --epochs=200 | tee {output_file_old}'
                )

                print(f"Running NEW: {cmd_new}")
                result1 = subprocess.run(cmd_new, shell=True)

                print(f"Running OLD: {cmd_old}")
                result2 = subprocess.run(cmd_old, shell=True)

                # Khởi tạo mặc định
                best_train_loss = best_train_accuracy = best_test_loss = best_test_accuracy = -1
                time_extract_shapelet = num_candidate = -1
                best_train_loss_new = best_train_accuracy_new = best_test_loss_new = best_test_accuracy_new = -1
                time_extract_shapelet_new = num_candidate_new = num_candidate_after = -1

                try:
                    with open("result_train2/" + result_file_old, "r") as f:
                        lines = f.readlines()
                        if len(lines) >= 6:
                            best_train_loss = float(lines[0].strip())
                            best_train_accuracy = float(lines[1].strip())
                            best_test_loss = float(lines[2].strip())
                            best_test_accuracy = float(lines[3].strip())
                            time_extract_shapelet = float(lines[4].strip())
                            num_candidate = float(lines[5].strip())
                except Exception as e:
                    print(f"⚠️ Không đọc được file cũ: {dataset}: {e}")

                try:
                    with open("result_train2/" + result_file_new, "r") as f:
                        lines = f.readlines()
                        if len(lines) >= 7:
                            best_train_loss_new = float(lines[0].strip())
                            best_train_accuracy_new = float(lines[1].strip())
                            best_test_loss_new = float(lines[2].strip())
                            best_test_accuracy_new = float(lines[3].strip())
                            time_extract_shapelet_new = float(lines[4].strip())
                            num_candidate_new = float(lines[5].strip())
                            num_candidate_after = float(lines[6].strip())
                except Exception as e:
                    print(f"⚠️ Không đọc được file mới: {dataset}: {e}")

                # if result1.returncode == 0 and result2.returncode == 0:
                success_log.append({
                    'Dataset': dataset,
                    'Size_MB': size,
                    'Num_Shapelet': num_shapelet,
                    'Window_Size': window_size,
                    'Extract_Time_old': round(time_extract_shapelet, 4),
                    'Extract_Time_new': round(time_extract_shapelet_new, 4),
                    'Final_Loss_train_old': best_train_loss,
                    'Final_Acc_train_old': best_train_accuracy,
                    'Final_Loss_test_old': best_test_loss,
                    'Final_Acc_test_old': best_test_accuracy,
                    'Final_Loss_train_new': best_train_loss_new,
                    'Final_Acc_train_new': best_train_accuracy_new,
                    'Final_Loss_test_new': best_test_loss_new,
                    'Final_Acc_test_new': best_test_accuracy_new,
                    'num_candidate_old': num_candidate,
                    'num_candidate_new': num_candidate_new,
                    'num_candidate_after_round1': num_candidate_after
                })

                append_processed_dataset(dataset)

                count += 1
                if count >= max_datasets:
                    print("Reached maximum of 30 datasets.")
                    break
        else:
            print(f"⚠️ Dataset '{dataset}' không có trong configs. Bỏ qua.")

    with open("success_log.csv", "w", newline='') as csvfile:
        fieldnames = [
            'Dataset', 'Size_MB', 'Num_Shapelet', 'Window_Size',
            'Extract_Time_old', 'Extract_Time_new',
            'Final_Loss_train_old', 'Final_Acc_train_old',
            'Final_Loss_test_old', 'Final_Acc_test_old',
            'Final_Loss_train_new', 'Final_Acc_train_new',
            'Final_Loss_test_new', 'Final_Acc_test_new',
            'num_candidate_old', 'num_candidate_new', 'num_candidate_after_round1'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(success_log)

if __name__ == "__main__":
    run_filtered_commands("folder_sizes.txt", "results/ppsn_vs_sota.csv", max_datasets=100)
