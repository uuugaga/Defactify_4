import os
import torch
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline  # Using Hugging Face's diffusers library

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Main folder path for images
data_folder = './data'
small_train_folder = 'small_train'
main_folder_path = os.path.join(data_folder, small_train_folder)
lare2_folder = 'LaRE2'
result_main_folder = os.path.join(data_folder, lare2_folder, small_train_folder)
os.makedirs(result_main_folder, exist_ok=True)

# Load Stable Diffusion encoder
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Extract only the encoder part
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

# Iterate through subfolders in the main folder
for sub_folder in os.listdir(main_folder_path):
    sub_folder_path = os.path.join(main_folder_path, sub_folder)
    result_sub_folder = os.path.join(result_main_folder, sub_folder)
    os.makedirs(result_sub_folder, exist_ok=True)

    if os.path.isdir(sub_folder_path):
        # Iterate through images in the subfolder
        for idx, filename in enumerate(os.listdir(sub_folder_path)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                print(f'Processing: {sub_folder}/{filename}')
                # Read and process the image
                img_path = os.path.join(sub_folder_path, filename)
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                image = image.to(torch.float16)

                # Extract the encoder's result
                with torch.no_grad():
                    encoder_result = vae.encode(image).latent_dist.sample()

                # Add noise
                noise = torch.randn_like(encoder_result).to(device)
                noisy_latent = encoder_result + noise

                # Perform one denoising step
                timesteps = torch.tensor([scheduler.num_train_timesteps - 1], device=device, dtype=torch.long)
                with torch.no_grad():
                    encoder_hidden_states = pipe.text_encoder(pipe.tokenizer("a photo", return_tensors="pt").input_ids.to(device))[0]
                    denoised_latent = unet(noisy_latent, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                # Save the result as an image
                denoised_latent_image = denoised_latent.squeeze().cpu().numpy()
                # Adjust data range to 0-255
                denoised_latent_image = (denoised_latent_image - denoised_latent_image.min()) / (denoised_latent_image.max() - denoised_latent_image.min()) * 255
                denoised_latent_image = denoised_latent_image.astype('uint8')
                # If the image is multi-channel, take the first channel
                if denoised_latent_image.ndim == 3:
                    denoised_latent_image = denoised_latent_image[0]
                result_image = Image.fromarray(denoised_latent_image)

                # Save the image to the corresponding result folder
                result_image.save(os.path.join(result_sub_folder, f"{filename}.png"))
