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
    fft_result = np.log(np.abs(fft_result) + 1)
    
    return fft_result

def process_image(args, img_path):
    if img_path.suffix in ['.png', '.jpg']:
        image = Image.open(img_path).convert('RGB')

        # Apply the cross-difference filter
        result_image = cross_difference_filter(image)
        result_image = result_image.resize((args.img_size, args.img_size), Image.NEAREST)

        # Apply FFT to the filtered image
        fft_image = apply_fft(result_image)

        return fft_image
    
    return None

def main(args):
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Create a list of tasks
    for class_name in data_path.iterdir():
        if class_name.is_dir():
            class_len = len(list(class_name.iterdir()))
            average_fft_image = np.zeros((args.img_size, args.img_size), dtype=np.float32)
            for img_path in tqdm(class_name.iterdir(), desc=f'Processing {class_name.name}', total=class_len, ncols=75):
                if img_path.suffix in ['.png', '.jpg']:
                    fft_image = process_image(args, img_path)
                    average_fft_image += fft_image / class_len

            average_fft_image = 1000 * average_fft_image
            average_fft_image = np.clip(average_fft_image, 0, 255)
            average_fft_image = Image.fromarray(np.uint8(average_fft_image))
            average_fft_image.save(results_path / f'{class_name.name}_average_fft.png')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='AI Model Experiment Controller')
    parser.add_argument('--data_path', type=str, default='../data/train', help='path to training dataset (default: ../data/train)')
    parser.add_argument('--results_path', type=str, default='./results', help='path to save experiment results (default: ./results)')
    parser.add_argument('--img_size', type=int, default=256, help='image size (default: 256)')

    args = parser.parse_args()

    main(args)

