import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import timm
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import sys
import cv2
import io
import random

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


def _apply_compression(img, quality):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def _add_gaussian_noise(tensor, sigma):
    noise = torch.randn(tensor.size()) * sigma
    return tensor + noise

class MultiFeatureDataset(Dataset):
    def __init__(self, args):
        self.features_dir = Path(args.train_features_path)
        self.original_images_dir = Path(args.train_path)
        self.features = args.features_selected
        self.categories = args.classes_list
        self.img_size = args.img_size
        self.binary = args.binary
        self.compression_quality = args.compression_quality
        self.crop_factor = args.crop_factor
        self.blur_sigma = args.blur_sigma
        self.noise_sigma = args.noise_sigma
        
        # Define transformations
        self.transform_rgb = self._get_transform_rgb()
        self.transform_gray = self._get_transform_gray()
        
        # Define label dictionary
        self.label_dict = self._create_label_dict()
        logging.info(f"Label dictionary: {self.label_dict}")

        # Gather samples
        self.samples = self._gather_samples()
        logging.info(f"Number of training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        category, file_paths = self.samples[idx]
        images = [self._load_image(path, category) for path in file_paths]       
        combined_image = torch.cat(images, dim=0)
        return combined_image, self.label_dict[category]

    def _get_transform_rgb(self):
        transforms_list = [transforms.Resize((self.img_size, self.img_size))]
        
        if self.compression_quality != 100:
            transforms_list.append(transforms.Lambda(lambda img: _apply_compression(img, self.compression_quality)))
        if self.crop_factor != 1.0:
            transforms_list.append(transforms.CenterCrop((int(self.img_size * self.crop_factor), int(self.img_size * self.crop_factor))))
            transforms_list.append(transforms.Resize((self.img_size, self.img_size)))
        if self.blur_sigma > 0:
            transforms_list.append(transforms.GaussianBlur(kernel_size=(5, 5), sigma=self.blur_sigma))
        
        transforms_list.append(transforms.ToTensor())
        
        if self.noise_sigma > 0:
            transforms_list.append(transforms.Lambda(lambda tensor: _add_gaussian_noise(tensor, random.uniform(0, self.noise_sigma))))

        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        return transforms.Compose(transforms_list)


    def _get_transform_gray(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _create_label_dict(self):
        if self.binary:
            return {category: 0 if 'coco' in category.lower() else 1 for category in self.categories}
        else:
            return {category: idx for idx, category in enumerate(self.categories)}

    def _gather_samples(self):
        samples = []
        for category in self.categories:
            file_name_dict = self._gather_file_names(category)
            for file_paths in file_name_dict.values():
                if len(file_paths) == len(self.features):
                    samples.append((category, file_paths))
                else:
                    logging.warning(f"Incomplete feature set for sample, skipping: {file_paths}")
        return samples

    def _gather_file_names(self, category):
        file_name_dict = {}
        for feature in self.features:
            image_paths = self._get_image_paths(category, feature)
            for path in image_paths:
                if path.stem not in file_name_dict:
                    file_name_dict[path.stem] = [path]
                else:
                    file_name_dict[path.stem].append(path)
        return file_name_dict

    def _get_image_paths(self, category, feature):
        if feature == "rgb":
            return list((self.original_images_dir / category).glob('*.[jp][np][g]'))
        else:
            return list((self.features_dir / category / feature).glob('*.[jp][np][g]'))

    def _load_image(self, file_path, category):
        try:
            if Path(self.original_images_dir / category) in file_path.parents:
                image = Image.open(file_path).convert('RGB')
                transform = self._get_transform_rgb()
                return transform(image)
            else:
                image = Image.open(file_path).convert('L')
                return self.transform_gray(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {file_path}: {e}")
    

class ValidationDataset(Dataset):
    def __init__(self, args):
        self.features_dir = Path(args.val_features_path)
        self.original_images_dir = Path(args.val_path)
        self.labels_file = Path(args.labels_file)
        self.img_size = args.img_size
        self.binary = args.binary
        self.features = args.features_selected
        self.compression_quality = args.compression_quality
        self.crop_factor = args.crop_factor
        self.blur_sigma = args.blur_sigma
        self.noise_sigma = args.noise_sigma
        
        # Define transformations
        self.transform_rgb = self._get_transform_rgb()
        self.transform_gray = self._get_transform_gray()
        
        # Load labels from .xlsx file
        self.labels_df = pd.read_excel(self.labels_file)
        logging.info(f"Loaded {len(self.labels_df)} validation labels from {self.labels_file}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        sample_id = row['Index']
        label = row['Label_A'] if self.binary else row['Label_B']
        images = [self._load_image(sample_id, feature) for feature in self.features]
        combined_image = torch.cat(images, dim=0)
        return combined_image, label

    def _get_transform_rgb(self):
        transforms_list = [transforms.Resize((self.img_size, self.img_size))]
        
        if self.compression_quality != 100:
            transforms_list.append(transforms.Lambda(lambda img: _apply_compression(img, self.compression_quality)))
        if self.crop_factor != 1.0:
            transforms_list.append(transforms.CenterCrop((int(self.img_size * self.crop_factor), int(self.img_size * self.crop_factor))))
            transforms_list.append(transforms.Resize((self.img_size, self.img_size)))
        if self.blur_sigma > 0:
            transforms_list.append(transforms.GaussianBlur(kernel_size=(5, 5), sigma=self.blur_sigma))
        
        transforms_list.append(transforms.ToTensor())
        
        if self.noise_sigma > 0:
            transforms_list.append(transforms.Lambda(lambda tensor: _add_gaussian_noise(tensor, self.noise_sigma)))
        
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        return transforms.Compose(transforms_list)

    def _get_transform_gray(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _load_image(self, sample_id, feature):
        if feature == "rgb":
            image_path = self._get_image_path(sample_id, self.original_images_dir)
            image = Image.open(image_path).convert('RGB')
            return self.transform_rgb(image)
        else:
            image_path = self._get_image_path(sample_id, self.features_dir / feature)
            image = Image.open(image_path).convert('L')
            return self.transform_gray(image)

    def _get_image_path(self, sample_id, directory):
        jpg_path = directory / f"{sample_id}.jpg"
        png_path = directory / f"{sample_id}.png"
        if jpg_path.exists():
            return jpg_path
        elif png_path.exists():
            return png_path
        else:
            raise FileNotFoundError(f"Image {sample_id} not found in {directory}")
    
class InferenceDataset(Dataset):
    def __init__(self, args):
        self.features_dir = Path(args.inference_features_path)
        self.original_images_dir = Path(args.inference_path)
        self.labels_file = Path(args.labels_file)
        self.img_size = args.img_size
        self.binary = args.binary
        self.features = args.features_selected
        self.compression_quality = args.compression_quality
        self.crop_factor = args.crop_factor
        self.blur_sigma = args.blur_sigma
        self.noise_sigma = args.noise_sigma
        
        # Define transformations
        self.transform_rgb = self._get_transform_rgb()
        self.transform_gray = self._get_transform_gray()

        # Load labels from .xlsx file
        self.labels_df = pd.read_excel(self.labels_file)
        logging.info(f"Loaded {len(self.labels_df)} testing labels from {self.labels_file}")

    def __len__(self):
        return len(list(self.original_images_dir.glob('*.[jp][np][g]')))

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        sample_id = row['Index']
        # label = row['Label_A'] if self.binary else row['Label_B']
        images = [self._load_image(sample_id, feature) for feature in self.features]
        combined_image = torch.cat(images, dim=0)
        return combined_image, sample_id

    def _get_transform_rgb(self):
        transforms_list = [transforms.Resize((self.img_size, self.img_size))]
        
        if self.compression_quality != 100:
            transforms_list.append(transforms.Lambda(lambda img: _apply_compression(img, self.compression_quality)))
        if self.crop_factor != 1.0:
            transforms_list.append(transforms.CenterCrop((int(self.img_size * self.crop_factor), int(self.img_size * self.crop_factor))))
            transforms_list.append(transforms.Resize((self.img_size, self.img_size)))
        if self.blur_sigma > 0:
            transforms_list.append(transforms.GaussianBlur(kernel_size=(5, 5), sigma=self.blur_sigma))
        
        transforms_list.append(transforms.ToTensor())
        
        if self.noise_sigma > 0:
            transforms_list.append(transforms.Lambda(lambda tensor: _add_gaussian_noise(tensor, self.noise_sigma)))
        
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        return transforms.Compose(transforms_list)

    def _get_transform_gray(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _load_image(self, sample_id, feature):
        if feature == "rgb":
            image_path = self._get_image_path(sample_id, self.original_images_dir)
            image = Image.open(image_path).convert('RGB')
            # image = np.array(image)
            # image = image[..., ::-1]
            # image = Image.fromarray(image)
            return self.transform_rgb(image)
        else:
            image_path = self._get_image_path(sample_id, self.features_dir / feature)
            image = Image.open(image_path).convert('L')
            return self.transform_gray(image)

    def _get_image_path(self, sample_id, directory):
        jpg_path = directory / f"{sample_id}.jpg"
        png_path = directory / f"{sample_id}.png"
        if jpg_path.exists():
            return jpg_path
        elif png_path.exists():
            return png_path
        else:
            raise FileNotFoundError(f"Image {sample_id} not found in {directory}")


class EfficientNetV2S(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EfficientNetV2S, self).__init__()

        self.effnet = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True)

        self.effnet.conv_stem = nn.Conv2d(
            in_channels=input_channels,  # Set the number of input channels (e.g., 5 for RGB + depth + infrared)
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
    
        self.effnet.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.effnet(x)
    
def plot_confusion_matrix(y_true, y_pred, classes, results_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    Path(results_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(results_path) / 'confusion_matrix.png')
    plt.close()

            
def train(args):
    train_dataset = MultiFeatureDataset(args)
    val_dataset = ValidationDataset(args)

    # create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Get the number of channels in an image sample
    num_channels = val_dataset[0][0].shape[0]
    logging.info(f'Number of channels in a sample: {num_channels}')

    # create model, loss function, and optimizer
    model = EfficientNetV2S(input_channels=num_channels, num_classes=2 if args.binary else len(args.classes_list)).to(args.device).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    best_f1_score = 0.0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc=f'Training Epoch: [{epoch + 1}/{args.epochs}]', ncols=75, leave=False):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_avg_loss = running_loss / len(train_dataloader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc=f'Validation Epoch: [{epoch + 1:2}/{args.epochs}]', ncols=75, leave=False):
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total

        # Calculate macro F1 score
        if args.binary:
            f1_score = sklearn.metrics.f1_score(all_labels, all_predictions, average='binary')
        else:
            f1_score = sklearn.metrics.f1_score(all_labels, all_predictions, average='macro')

        logging.info(f'Epoch: [{epoch + 1}/{args.epochs}], Loss: {train_avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1_score:.4f}')

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            if args.binary:
                model_path = Path(args.model_path).with_name(f'binary_best_model.pth')
            else:
                model_path = Path(args.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logging.info(f'Best model saved with F1 score: {best_f1_score:.4f}')

            # Plot confusion matrix
            classes = ['coco', 'other'] if args.binary else train_dataset.categories
            plot_confusion_matrix(all_labels, all_predictions, classes, args.results_path)

def evaluate(args):
    # create data loaders
    val_dataset = ValidationDataset(args)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Get the number of channels in an image sample
    num_channels = val_dataset[0][0].shape[0]
    logging.info(f'Number of channels in a sample: {num_channels}')
    
    model = EfficientNetV2S(input_channels=num_channels, num_classes=2 if args.binary else len(args.classes_list)).to(args.device).to(args.device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True), strict=True)
    model.eval()
    
    all_predictions = []
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader, desc='Evaluating', ncols=75, leave=False):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logging.info(f'Accuracy: {accuracy:.2f}%')

    
    # write predictions to .csv file
    df = pd.DataFrame({'Index': val_dataset.labels_df['Index'], 'Predicted': all_predictions})
    Path(args.results_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.results_path) / 'predictions.csv', index=False)

def inference(args):
    test_dataset = InferenceDataset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Get the number of channels in an image sample
    num_channels = test_dataset[0][0].shape[0]
    logging.info(f'Number of channels in a sample: {num_channels}')

    model = EfficientNetV2S(input_channels=num_channels, num_classes=2 if args.binary else len(args.classes_list)).to(args.device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True), strict=True)
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for images, _ in tqdm(test_dataloader, desc='Inference', ncols=75, leave=False):
            images = images.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())

    # write predictions to .csv file
    task1_predictions = [0 if pred == 0 else 1 for pred in all_predictions]
    df = pd.DataFrame({'index': test_dataset.labels_df['Index'], 'caption':test_dataset.labels_df['Caption'],  'Label_A': task1_predictions,'Label_B': all_predictions})
    Path(args.results_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.results_path) / 'inference_predictions.csv', index=False)
    df.to_json(Path(args.results_path) / 'answer.json', orient='records', indent=4)

    # zip the json file
    import zipfile
    with zipfile.ZipFile(Path(args.results_path) / 'answer.zip', 'w') as z:
        z.write(Path(args.results_path) / 'answer.json', 'answer.json')


def grad_cam(args):
    # Create data loaders
    val_dataset = ValidationDataset(args)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Generate class dictionary
    class_dict = {idx: category for idx, category in enumerate(args.classes_list)} if not args.binary else {0: category if 'coco' in category.lower() else 1 for category in args.classes_list}

    # Get the number of channels in an image sample
    num_channels = val_dataset[0][0].shape[0]
    logging.info(f'Number of channels in a sample: {num_channels}')

    # Select samples per class
    samples_per_class = 5
    samples_dict = {i: [] for i in range(len(args.classes_list))}

    for image, label in val_dataloader:
        image, label = image[0], label[0].item()
        if len(samples_dict[label]) < samples_per_class:
            samples_dict[label].append(image)
        if all(len(samples) == samples_per_class for samples in samples_dict.values()):
            break

    # Load the model
    model = EfficientNetV2S(input_channels=num_channels, num_classes=2 if args.binary else len(args.classes_list)).to(args.device)
    target_layers = [model.effnet.blocks[-1]]

    # Initialize GradCAM once
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        for label, images in samples_dict.items():
            for idx, image in enumerate(images):
                input_tensor = image.unsqueeze(0).to(args.device)
                targets = [ClassifierOutputTarget(label)]

                # Generate Grad-CAM heatmap
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                
                # Convert image for visualization
                rgb_img = (image.cpu().numpy().squeeze().transpose((1, 2, 0)) * 0.5 + 0.5).clip(0, 1)
                rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                rgb_img_uint8 = (rgb_img_bgr * 255).astype(np.uint8)

                # Create visualization
                visualization = show_cam_on_image(rgb_img_bgr, grayscale_cam, use_rgb=True)
                
                # Save visualization and original image
                output_dir = Path(args.results_path) / 'gradcam' / class_dict[label]
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_dir / f'{idx}.png'), visualization)
                cv2.imwrite(str(output_dir / f'{idx}_original.png'), rgb_img_uint8)

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AI Model Experiment Controller')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training (default: cuda:0)')
    parser.add_argument('--train_path', type=str, default='../data/train', help='path to training dataset (default: ../data/train)')
    parser.add_argument('--val_path', type=str, default='../data/val', help='path to original images dataset (default: ../data/val)')
    parser.add_argument('--labels_file', type=str, default='../data/val/Validation_labels.xlsx', help='path to labels file (default: ../data/val/Validation_labels.xlsx)')
    parser.add_argument('--train_features_path', type=str, default='../data/train_results', help='path to features dataset (default: ../data/features)')
    parser.add_argument('--val_features_path', type=str, default='../data/val_results', help='path to features dataset (default: ../data/features)')
    parser.add_argument('--features_selected', type=str, nargs='+', default=["rgb", "error", "frequency"], help='list of features to use (default: ["error", "frequency"])')
    parser.add_argument('--classes_list', type=str, nargs='+', default=["coco_image", "sd21_image", "sdxl_image", "sd3_image", "dalle_image", "midjourney_image"], help='list of class names to use (default: "coco_image", "sd3_image", "sd21_image", "sdxl_image", "dalle_image", "midjourney_image")')
    parser.add_argument('--results_path', type=str, default='./results', help='path to save experiment results (default: ./results)')
    parser.add_argument('--img_size', type=int, default=512, help='image size (default: 512)')
    parser.add_argument('--binary', action='store_true', help='flag to enable binary classification')
    parser.add_argument('--evaluate', action='store_true', help='flag to enable evaluation mode')
    parser.add_argument('--evaluate_path', default='../data/val', help='path to evaluate images dataset (default: ../data/val)')
    parser.add_argument('--inference', action='store_true', help='flag to enable inference mode')
    parser.add_argument('--inference_path', default='../data/test', help='path to inference images dataset (default: ../data/test)')
    parser.add_argument('--inference_features_path', default='../data/test_results', help='path to features dataset (default: ../data/features)')
    parser.add_argument('--grad_cam', action='store_true', help='flag to enable Grad-CAM mode')
    parser.add_argument('--num_workers', type=int, default=8, help='number of worker threads for data loading (default: 8)')
    parser.add_argument('--model_path', type=str, default='./results/best_model.pth', help='path to save trained model (default: ./results/model.pth)')
    parser.add_argument('--compression_quality', type=int, default=100, help='JPEG compression quality (default: 100)')
    parser.add_argument('--crop_factor', type=float, default=1.0, help='center crop factor (default: 1.0)')
    parser.add_argument('--blur_sigma', type=float, default=0.0, help='Gaussian blur sigma (default: 0.0)')
    parser.add_argument('--noise_sigma', type=float, default=0.0, help='Gaussian noise sigma (default: 0.0)')
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args)
    elif args.inference:
        inference(args)
    elif args.grad_cam:
        grad_cam(args)
    else:
        train(args)
