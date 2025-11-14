import os
from PIL import Image

def convert_images_to_jpg(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Full path to the file
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image and not already a JPG
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jfif', '.jpeg', '.bmp', '.gif')):
            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Convert image to RGB (to avoid errors with transparency)
                    img = img.convert("RGB")
                    
                    # Create the new file path with JPG extension
                    new_file_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.jpg")
                    
                    # Save the image as JPG
                    img.save(new_file_path, "JPEG")
                    
                    # Remove the original file
                    os.remove(file_path)
                    print(f"Converted and removed: {filename}")
                    
            except Exception as e:
                print(f"Error converting {filename}: {e}")

# Specify your folder path here
folder_path = "path/to/your/folder"
convert_images_to_jpg(folder_path)
