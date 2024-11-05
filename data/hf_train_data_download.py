import os
from datasets import load_dataset
from PIL import Image
import pandas as pd

ds = load_dataset("NasrinImp/Defactify4_Train")

base_dir = "train"
image_fields = ['coco_image', 'sd3_image', 'sd21_image', 'sdxl_image', 'dalle_image', 'midjourney_image']

os.makedirs(base_dir, exist_ok=True)
for field in image_fields:
    os.makedirs(os.path.join(base_dir, field), exist_ok=True)

captions_data = []

for i, example in enumerate(ds['train']):
    caption = example['caption']
    for field in image_fields:
        img = example[field]
        img_filename = f"image_{i}.png"
        img_path = os.path.join(base_dir, field, img_filename)
        img.save(img_path)
    # captions_data.append({'Index': i, 'Caption': caption})
    if (i + 1) % 100 == 0:
        print(f"Saved {i + 1} images...")

# captions_df = pd.DataFrame(captions_data)
# captions_df.to_excel(os.path.join(base_dir, 'captions.xlsx'), index=False)

print("All images and captions have been saved successfully!")
