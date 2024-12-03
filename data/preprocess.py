import cv2
import os
from tqdm import tqdm

def fix_exposure_in_folder(input_folder, output_folder, sigma_s=10, sigma_r=0.15):
    """
    Fixes the exposure of all images in the input folder by applying auto-exposure using detailEnhance.
    
    Parameters:
    - input_folder (str): Path to the input folder containing images.
    - output_folder (str): Path to the output folder where processed images will be saved.
    - sigma_s (float): Spatial smoothness parameter (controls how much detail is preserved).
    - sigma_r (float): Contrast enhancement parameter (controls how much contrast is increased).
    """
    # Check if output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # sort by integer
    images_dir = os.listdir(input_folder)
    # remove .xlsx file
    images_dir = [x for x in images_dir if not x.endswith('.xlsx')]
    images_dir = sorted(images_dir, key=lambda x: int(x.split('.')[0]))
    # Loop through all files in the input folder
    for filename in tqdm(images_dir):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (can check file extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(input_path)

            if image is None:
                print(f"Error reading image {filename}, skipping...")
                continue

            # Apply detailEnhance to brighten the image (auto-exposure)
            enhanced_image = cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)

            # Save the enhanced image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, enhanced_image)

            # print(f"Processed and saved: {filename}")

# Specify the input and output folders
input_folder = "./test"  # Change this to your input folder path
output_folder = ".//test_fix"  # Change this to your output folder path

os.makedirs(output_folder, exist_ok=True)

# Call the function to process the images
fix_exposure_in_folder(input_folder, output_folder)
