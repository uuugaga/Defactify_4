import argparse
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch
# import torch.nn.functional as F
from torchvision.transforms import functional as F
import numpy as np
import os

def process_image(img, transform, args, results_path, subfolder=""):
    # img_transform = transform(img).unsqueeze(0).to(args.device)

    if args.crop:
        crop_path = results_path / subfolder / f'{Path(img.filename).stem}.png'
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        # center crop img of 0.8 size
        width, height = img.size
        new_width = int(width * 0.8)
        new_height = int(height * 0.8)
        crop_image = F.center_crop(img, (new_height, new_width))
        crop_image.save(crop_path)
    
    elif args.noise:
        noise_path = results_path / subfolder / f'{Path(img.filename).stem}.png'
        noise_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert PIL image to numpy array
        img_array = np.array(img)    
        # Generate noise
        noise = np.random.normal(0, 0.1, img_array.shape)
        
        # Add noise to the image
        noisy_img_array = img_array + noise
        
        # Clip the values to be in the valid range [0, 255] and convert back to uint8
        noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL image
        noisy_img = Image.fromarray(noisy_img_array)
    
        noisy_img.save(noise_path)
    
    elif args.compression:
        compression_path = results_path / subfolder / f'{Path(img.filename).stem}.png'
        compression_path.parent.mkdir(parents=True, exist_ok=True)
        # compress img
        img.save(compression_path, quality=80)

def read_images(data_path, results_path, transform, args):
    # Read data path folder
    for class_name in data_path.iterdir():
        if class_name.is_dir():
            class_len = len(list(class_name.iterdir()))
            for img_path in tqdm(class_name.iterdir(), desc=f'Processing {class_name.name}', ncols=75, total=class_len):
                try:
                    img = Image.open(img_path)
                    process_image(img, transform, args, results_path, class_name.name)
                except (OSError, IOError) as e:
                    print(f"Error loading image {img_path}: {e}")


def data_augment(args):

    # pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_id).to(args.device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    data_path = args.data_path
    if args.crop:
        print("Crop generation enabled")
        results_path = data_path + '_crop'
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        data_path = Path(data_path)
        print(f"Results will be saved in {results_path}")
        read_images(data_path, results_path, transform, args)

    if args.noise:
        print("Noise generation enabled")
        results_path = data_path + '_noise'
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        data_path = Path(data_path)
        print(f"Results will be saved in {results_path}")
        read_images(data_path, results_path, transform, args)
    
    if args.compression:
        print("Compression generation enabled")
        results_path = data_path + '_compression'
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        data_path = Path(data_path)
        print(f"Results will be saved in {results_path}")
        read_images(data_path, results_path, transform, args)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Model Experiment Controller')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training (default: cuda:0)')
    parser.add_argument('--data_path', type=str, default='../data/train', help='path to training dataset (default: ../data/train)')
    # parser.add_argument('--results_path', type=str, default='../data/train', help='path to save experiment results (default: ./results)')
    parser.add_argument('--img_size', type=int, default=512, help='image size (default: 512)')
    parser.add_argument('--crop', action='store_true', help='flag to enable crop generation')
    parser.add_argument('--noise', action='store_true', help='flag to enable noise generation')
    parser.add_argument('--compression', action='store_true', help='flag to enable compression generation')
    parser.add_argument('--flat_structure', action='store_true', help='flag to indicate flat folder structure without classes')
    args = parser.parse_args()

    data_augment(args)