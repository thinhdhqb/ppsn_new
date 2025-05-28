import os

def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def save_sorted_folder_sizes_to_txt(base_path, output_file):
    folder_sizes = []

    # Duyệt qua tất cả các thư mục con
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            size_bytes = get_folder_size(folder_path)
            size_mb = size_bytes / (1024 * 1024)
            folder_sizes.append((folder, size_mb))

    # Sắp xếp theo thứ tự bảng chữ cái
    folder_sizes.sort(key=lambda x: x[0].lower())

    # Ghi ra file
    with open(output_file, 'w') as f:
        for folder, size_mb in folder_sizes:
            f.write(f"{folder}: {size_mb:.2f} MB\n")
    print(f"Saved folder sizes to {output_file}")

# Replace with your target path and output file
base_directory = "dataset/UCRArchive_2018"
output_txt = "folder_sizes.txt"

save_sorted_folder_sizes_to_txt(base_directory, output_txt)
