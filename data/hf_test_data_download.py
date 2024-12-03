import os
from datasets import load_dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Load the dataset
ds = load_dataset("NasrinImp/Final_defactify_test_new")

# Get the directory where the dataset is saved
dataset_dir = os.path.dirname(
    os.path.abspath(__file__)
)  # Get current script directory (assuming script is in the same directory as the dataset)

# Create directories for saving images and captions inside the dataset's directory
base_dir = os.path.join(dataset_dir, "test")
os.makedirs(base_dir, exist_ok=True)

# List to hold the captions and their indices
captions_data = []

# Loop through each example in the dataset
for i, example in enumerate(tqdm(ds["train"])):
    # Extract the image and caption
    img = example["image"]  # Assuming the image column is named 'image'
    caption = example["caption"]

    # Create a filename for the image
    img_filename = f"{i}.png"
    img_path = os.path.join(base_dir, img_filename)

    # Save the image to the file system
    img.save(img_path)

    # Save the caption data for later use
    captions_data.append({"Index": i, "Caption": caption})

    # # Print progress every 50 images
    # if (i + 1) % 50 == 0:
    #     print(f"Saved {i + 1} images...")

# Save the captions data to an Excel file
captions_df = pd.DataFrame(captions_data)
captions_df.to_excel(os.path.join(base_dir, "captions.xlsx"), index=False)

print(f"All images and captions have been saved successfully in {base_dir}!")
