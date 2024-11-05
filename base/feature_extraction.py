import argparse
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

def latent_to_np(img):
    return (
        (
            (img / 2 + 0.5)
            .clamp(0, 1)
            .squeeze()
            .permute(1, 2, 0)
            * 255
        )
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

def process_image(img, transform, pipeline, args, results_path, subfolder=""):
    img_transform = transform(img).unsqueeze(0).to(args.device)

    if args.reconstruction:
        with torch.no_grad():
            latent = pipeline.vae.encode(img_transform).latent_dist.sample()
            reconsturcted_img = pipeline.vae.decode(latent).sample
            reconstructed_img_np = latent_to_np(reconsturcted_img)
            result_path = results_path / subfolder / f'reconstruction' / f'{Path(img.filename).stem}.png'
            result_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reconstructed_img_np).save(result_path)

    if args.reconstruction and args.error:
        error_map = torch.abs(img_transform - reconsturcted_img)
        error_map_np = latent_to_np(error_map)
        error_map_path = results_path / subfolder / f'error' / f'{Path(img.filename).stem}.png'
        error_map_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(error_map_np).save(error_map_path)

    if args.latent:
        latent_path = results_path / subfolder / f'latent' / f'{Path(img.filename).stem}.pt'
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent, latent_path)

    if args.frequency:
        f = np.fft.fft2(np.array(img.resize((args.img_size, args.img_size)).convert("L")))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift + 1e-8))
        magnitude_spectrum_img = Image.fromarray(magnitude_spectrum.astype(np.uint8))
        frequency_path = results_path / subfolder / f'frequency' / f'{Path(img.filename).stem}.png'
        frequency_path.parent.mkdir(parents=True, exist_ok=True)
        magnitude_spectrum_img.save(frequency_path)

def feature_extraction(args):

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_id).to(args.device)

    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in {results_path}")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Read data path folder
    if args.flat_structure:
        img_paths = list(data_path.glob('*.[jp][np][g]'))
        print(f"Found {len(img_paths)} images")
        for img_path in tqdm(img_paths, desc=f'Processing images', ncols=75, total=len(img_paths)):
            try:
                img = Image.open(img_path)
                process_image(img, transform, pipeline, args, results_path)
            except (OSError, IOError) as e:
                print(f"Error loading image {img_path}: {e}")
    else:
        for class_name in data_path.iterdir():
            if class_name.is_dir():
                class_len = len(list(class_name.iterdir()))
                for img_path in tqdm(class_name.iterdir(), desc=f'Processing {class_name.name}', ncols=75, total=class_len):
                    try:
                        img = Image.open(img_path)
                        process_image(img, transform, pipeline, args, results_path, class_name.name)
                    except (OSError, IOError) as e:
                        print(f"Error loading image {img_path}: {e}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AI Model Experiment Controller')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training (default: cuda:0)')
    parser.add_argument('--data_path', type=str, default='../data/train', help='path to training dataset (default: ../data/train)')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='diffusers pipeline model ID (default: runwayml/stable-diffusion-v1-5)')
    parser.add_argument('--results_path', type=str, default='./results', help='path to save experiment results (default: ./results)')
    parser.add_argument('--img_size', type=int, default=512, help='image size (default: 512)')
    parser.add_argument('--reconstruction', action='store_true', help='flag to enable reconstruction generation')
    parser.add_argument('--delta', action='store_true', help='flag to enable delta generation')
    parser.add_argument('--error', action='store_true', help='flag to enable error generation')
    parser.add_argument('--frequency', action='store_true', help='flag to enable frequency generation')
    parser.add_argument('--latent', action='store_true', help='flag to enable latent generation')
    parser.add_argument('--flat_structure', action='store_true', help='flag to indicate flat folder structure without classes')
    args = parser.parse_args()

    feature_extraction(args)