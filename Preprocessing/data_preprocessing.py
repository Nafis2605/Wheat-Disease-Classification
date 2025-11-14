# %%
# created by Shahana Shultana for corrupted file removing and data pre-processing

#removing corrupted file

import os
from PIL import Image

def remove_corrupted_images(directory):
    # Loop through each class folder in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct full file path
            file_path = os.path.join(root, file)
            try:
                # Try opening the image file
                with Image.open(file_path) as img:
                    img.verify()  # Verify the integrity of the image
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image found and removed: {file_path}")
                os.remove(file_path)  # Remove corrupted image

# Directories for train, test, and validation folders
directories = ['./all/train', './all/test', './all/val']

# Remove corrupted images in all specified directories
for directory in directories:
    remove_corrupted_images(directory)


# %%
#check any corrupted file exist or not
import os
from PIL import Image

def check_for_remaining_corrupted_images(directory):
    corrupted_count = 0
    # Loop through each class folder in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct full file path
            file_path = os.path.join(root, file)
            try:
                # Try opening the image file to check if it's corrupted
                with Image.open(file_path) as img:
                    img.verify()  # Verify the integrity of the image
            except (IOError, SyntaxError) as e:
                # If it's corrupted, count it
                corrupted_count += 1
    return corrupted_count

# Directories for train, test, and validation folders
directories = ['./all/train', './all/test', './all/val']

# Check for remaining corrupted images in all specified directories
for directory in directories:
    remaining_corrupted = check_for_remaining_corrupted_images(directory)
    
    if remaining_corrupted > 0:
        print(f"Remaining corrupted images in {directory}: {remaining_corrupted}")
    else:
        print(f"No corrupted images remaining in {directory}")


# %%
#per class count
import os
import matplotlib.pyplot as plt
from torchvision import datasets
import numpy as np

# Define paths to your datasets
data_paths = {
    'train': './all/train',
    'validation': './all/val',
    'test': './all/test'
}

# Function to count images per class
def count_images(data_path):
    dataset = datasets.ImageFolder(data_path)
    class_counts = {dataset.classes[i]: 0 for i in range(len(dataset.classes))}
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1
    return class_counts

# Plotting function with different colors for each bar
def plot_class_distribution(data_counts, title, color_palette):
    fig, ax = plt.subplots()
    classes = list(data_counts.keys())
    counts = list(data_counts.values())
    # Generate a color array using the provided palette and ensuring one color per class
    colors = plt.cm.get_cmap(color_palette, len(classes))(np.arange(len(classes)))

    ax.bar(classes, counts, color=colors)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Images')
    ax.set_title(title)
    ax.set_xticklabels(classes, rotation=45, ha='right')

    # Print class counts
    print(f"\n{title} - Class Counts:")
    for cls, count in data_counts.items():
        print(f"{cls}: {count}")

    plt.tight_layout()
    plt.show()

# Color palettes for different datasets
color_palettes = {
    'train': 'viridis',
    'validation': 'cividis',
    'test': 'plasma'
}

# Get counts and plot for each dataset
for key, path in data_paths.items():
    counts = count_images(path)
    plot_class_distribution(counts, f'{key.capitalize()} Set Class Distribution', color_palettes[key])

# %%
# Import necessary libraries
import os
import numpy as np
from PIL import Image
from torchvision import datasets

# Define paths for the dataset and output directory for NumPy files
data_dir = './all'  # Base directory containing train, test, and valid subfolders
output_dir = './numpy_all_data'  # Directory to store processed NumPy files
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

def convert_to_numpy_no_transform(data_folder, output_prefix):
    # Load images from the directory using ImageFolder
    dataset = datasets.ImageFolder(root=data_folder)
    images = []
    labels = []

    # Iterate through the dataset to convert images and labels to NumPy arrays
    for img_path, label in dataset.imgs:
        img = Image.open(img_path).convert("RGB")  # Ensure all images are in RGB format
        img_resized = img.resize((224, 224))  # Resize image to 224x224
        img_array = np.array(img_resized)  # Convert the image to a NumPy array
        images.append(img_array)
        labels.append(label)

    # Stack images and labels into NumPy arrays
    images = np.stack(images)
    labels = np.array(labels)

    # Save images and labels to .npy files
    np.save(os.path.join(output_dir, f'{output_prefix}_images.npy'), images)
    np.save(os.path.join(output_dir, f'{output_prefix}_labels.npy'), labels)
    print(f'Saved {output_prefix}_images.npy and {output_prefix}_labels.npy to {output_dir}')

# Paths to train, test, and valid subfolders
train_folder = os.path.join(data_dir, 'train')
valid_folder = os.path.join(data_dir, 'val')
test_folder = os.path.join(data_dir, 'test')

# Convert each dataset (train, valid, test) to NumPy arrays and save them
convert_to_numpy_no_transform(train_folder, 'train')
convert_to_numpy_no_transform(valid_folder, 'val')
convert_to_numpy_no_transform(test_folder, 'test')

# Load the saved NumPy files
train_images = np.load(os.path.join(output_dir, 'train_images.npy'))
train_labels = np.load(os.path.join(output_dir, 'train_labels.npy'))

print(f'Train Images Shape: {train_images.shape}')
print(f'Train Labels Shape: {train_labels.shape}')

# Display a sample image (Optional)
import matplotlib.pyplot as plt

# Select a sample image and display it
sample_image = train_images[0]  # Image is already in HWC format (height, width, channels)
plt.imshow(sample_image)
plt.title(f'Sample Image - Label: {train_labels[0]}')
plt.axis('off')
plt.show()


# %%
# Import necessary libraries
import os
import numpy as np
from PIL import Image
from torchvision import datasets

# Define paths for the dataset and output directory for NumPy files
data_dir = './all1'  # Base directory containing train, test, and valid subfolders
output_dir = './numpy2/new'  # Directory to store processed NumPy files
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

def convert_to_numpy_no_transform(data_folder, output_prefix):
    # Load images from the directory using ImageFolder
    dataset = datasets.ImageFolder(root=data_folder)
    images = []
    labels = []

    # Iterate through the dataset to convert images and labels to NumPy arrays
    for img_path, label in dataset.imgs:
        img = Image.open(img_path).convert("RGB")  # Ensure all images are in RGB format
        img_resized = img.resize((224, 224))  # Resize image to 224x224
        img_array = np.array(img_resized)  # Convert the image to a NumPy array
        images.append(img_array)
        labels.append(label)

    # Stack images and labels into NumPy arrays
    images = np.stack(images)
    labels = np.array(labels)

    # Save images and labels to .npy files
    np.save(os.path.join(output_dir, f'{output_prefix}_images.npy'), images)
    np.save(os.path.join(output_dir, f'{output_prefix}_labels.npy'), labels)
    print(f'Saved {output_prefix}_images.npy and {output_prefix}_labels.npy to {output_dir}')

# Paths to train, test, and valid subfolders
train_folder = os.path.join(data_dir, 'train')
valid_folder = os.path.join(data_dir, 'val')
test_folder = os.path.join(data_dir, 'test')

# Convert each dataset (train, valid, test) to NumPy arrays and save them
convert_to_numpy_no_transform(train_folder, 'train')
convert_to_numpy_no_transform(valid_folder, 'val')
convert_to_numpy_no_transform(test_folder, 'test')

# Load the saved NumPy files
train_images = np.load(os.path.join(output_dir, 'train_images.npy'))
train_labels = np.load(os.path.join(output_dir, 'train_labels.npy'))

print(f'Train Images Shape: {train_images.shape}')
print(f'Train Labels Shape: {train_labels.shape}')

# Display a sample image (Optional)
import matplotlib.pyplot as plt

# Select a sample image and display it
sample_image = train_images[0]  # Image is already in HWC format (height, width, channels)
plt.imshow(sample_image)
plt.title(f'Sample Image - Label: {train_labels[0]}')
plt.axis('off')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_images(image_folder, indices):
    # List all files in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    selected_images = [image_files[i] for i in indices]
    return selected_images

def process_and_display_images(image_files):
    fig, axes = plt.subplots(len(image_files), 2, figsize=(10, 5 * len(image_files)))  # 2 columns for before and after

    for i, image_path in enumerate(image_files):
        # Load the image
        img = Image.open(image_path).convert('RGB')

        # Display the original image
        ax = axes[i, 0]
        ax.imshow(img)
        ax.set_title(f'Original Image {i+1}')
        ax.axis('off')

        # Process the image
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0  # Normalize to range [0, 1]

        # Display the processed image
        ax = axes[i, 1]
        ax.imshow(img_array)
        ax.set_title(f'Processed Image {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Define the directory containing your images
image_folder = './all1/train/brown rust/'

# Specify the indices of the images to display
indices = [12, 13, 14]  # Indices for the 2nd, 3rd, and 700th images, adjusting for 0-based index

# Load images by indices
image_files = load_images(image_folder, indices)

# Process and display the images
process_and_display_images(image_files)



