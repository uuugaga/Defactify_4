from PIL import Image
import numpy as np
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cross_difference_filter(image):
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Initialize output image with the same size and dtype as the original image
    filtered_image = np.zeros(image_array.shape, dtype=image_array.dtype)
    
    # Split the image into channels
    channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
    filtered_channels = []

    # Calculate cross-difference value for each channel using vectorized operations
    for channel in channels:
        # Calculate cross-difference using slicing and vectorized operations
        value = abs(
            np.int16(channel[:-1, :-1]) + np.int16(channel[1:, 1:])
            - np.int16(channel[:-1, 1:]) - np.int16(channel[1:, :-1])
        )
        filtered_channel = np.clip(value, 0, 255).astype(np.uint8)
        # Pad the filtered channel to match the original size
        filtered_channel = np.pad(filtered_channel, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        filtered_channels.append(filtered_channel)
    
    # Merge the filtered channels back together
    filtered_image = np.stack(filtered_channels, axis=-1)
    
    return Image.fromarray(filtered_image)


def apply_fft(image):
    # Convert image to grayscale
    image_gray = image.convert('L')
    image_array = np.array(image_gray)
    
    # Apply FFT and normalize by the size
    fft_result = np.fft.fftshift(np.fft.fft2(image_array)) / (image_array.shape[0] * image_array.shape[1])
    magnitude_spectrum = 500 * np.log(np.abs(fft_result) + 1)

    # Save the FFT result as an image
    magnitude_image = Image.fromarray(np.uint8(magnitude_spectrum))
    # magnitude_image = magnitude_image.resize((64, 64), Image.LANCZOS)

    
    return magnitude_image

def process_image(args):
    img_path, class_name, results_path = args
    if img_path.suffix in ['.png', '.jpg']:
        image = Image.open(img_path).convert('RGB')

        # Apply the cross-difference filter
        result_image = cross_difference_filter(image)
        result_image = result_image.resize((256, 256), Image.NEAREST)

        # Apply FFT to the filtered image
        fft_image = apply_fft(result_image)

        save_path = results_path / class_name / img_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fft_image.save(save_path)

def main(args):
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Create a list of tasks
    tasks = []
    for class_name in data_path.iterdir():
        if class_name.is_dir():
            for img_path in class_name.iterdir():
                if img_path.suffix in ['.png', '.jpg']:
                    tasks.append((img_path, class_name.name, results_path))

    # Process images in parallel
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc='Processing Images', ncols=75))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='AI Model Experiment Controller')
    parser.add_argument('--data_path', type=str, default='./data/train', help='path to training dataset (default: ../data/train)')
    parser.add_argument('--results_path', type=str, default='./results', help='path to save experiment results (default: ./results)')
    parser.add_argument('--img_size', type=int, default=512, help='image size (default: 512)')

    args = parser.parse_args()

    main(args)

