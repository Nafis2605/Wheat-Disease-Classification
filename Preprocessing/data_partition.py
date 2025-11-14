import os
import random
import shutil

# Define your dataset directory and the destination directories for train, val, and test
data_dir = '../../Downloads/Final_Dataset/Main/Combination/Yellow Rust'
train_dir = '../../Downloads/Final_Dataset/Main/Split/Yellow Rust/train'
val_dir = '../../Downloads/Final_Dataset/Main/Split/Yellow Rust/validation'
test_dir = '../../Downloads/Final_Dataset/Main/Split/Yellow Rust/test'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all files in the dataset directory
all_files = os.listdir(data_dir)
random.shuffle(all_files)  # Shuffle the files for random splitting

# Split the dataset
train_split = int(0.6 * len(all_files))
val_split = int(0.8 * len(all_files))

train_files = all_files[:train_split]
val_files = all_files[train_split:val_split]
test_files = all_files[val_split:]

# Function to move files to respective directories
def move_files(file_list, destination_dir):
    for file_name in file_list:
        src_path = os.path.join(data_dir, file_name)
        dest_path = os.path.join(destination_dir, file_name)
        shutil.move(src_path, dest_path)

# Move files to each set
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("Dataset split and moved successfully!")

def rename_files_in_dir(directory):
    files = sorted(os.listdir(directory))  # Sort files to ensure consistency
    for i, file_name in enumerate(files):
        file_extension = os.path.splitext(file_name)[1]  # Get the file extension
        new_name = f"{i+1:04}{file_extension}"  # Format with leading zeros, e.g., 0001, 0002
        src_path = os.path.join(directory, file_name)
        dest_path = os.path.join(directory, new_name)
        os.rename(src_path, dest_path)

# Define your directories
train_dir = '../../Downloads/Final_Dataset/Main/Split/Yellow Rust/train'
val_dir = '../../Downloads/Final_Dataset/Main/Split/Yellow Rust/validation'
test_dir = '../../Downloads/Final_Dataset/Main/Split/Yellow Rust/test'

# Rename files in each directory
rename_files_in_dir(train_dir)
rename_files_in_dir(val_dir)
rename_files_in_dir(test_dir)

print("Files renamed successfully!")