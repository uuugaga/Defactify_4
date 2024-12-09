from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from pathlib import Path
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import argparse
import sys
import pandas as pd
import joblib
import io
from torchvision import transforms
import os
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from tqdm import tqdm

import logging
from multiprocessing import Pool
from itertools import product
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

def get_augmented_images(image):
    """Generate augmented versions of the input image"""
    # Original image
    images = [image]
    
    # 1. Horizontal Flip
    flip = transforms.RandomHorizontalFlip(p=1.0)
    images.append(flip(image))
    
    # 2. Brightness Reduction
    brightness = transforms.ColorJitter(brightness=0.7)
    images.append(brightness(image))
    
    # 3. Gaussian Noise
    def add_gaussian_noise(image, mean=0., std=0.1):
        image_np = np.array(image)
        noise = np.random.normal(mean, std, image_np.shape)
        noisy_image = np.clip(image_np + noise * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)
    
    images.append(add_gaussian_noise(image))
    
    # 4. JPEG Compression
    def jpeg_compression(image, quality=75):
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    images.append(jpeg_compression(image))

    # 5. Brightness Reduction + Gaussian Noise
    images.append(brightness(add_gaussian_noise(image)))

    # 6. Brightness Reduction + JPEG Compression
    images.append(brightness(jpeg_compression(image)))

    # 7. Gaussian Noise + JPEG Compression
    images.append(add_gaussian_noise(jpeg_compression(image)))

    return images

class ImageFeatureExtractor:
    def __init__(self, original_images_dir=None, clip_model=None, mode=None):
        self.original_images_dir = Path(original_images_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.mode = mode

    def _get_image_paths(self, category=""):
        return list((self.original_images_dir / category).glob('*.[jp][np][g]'))
  
    def _load_image(self, file_path):
        try:
            return Image.open(file_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {file_path}: {e}")

    def extract_features(self, image_paths):
        features = []
        indices = []
        count = 0
        for image_path in tqdm(image_paths, total=len(image_paths), desc="Extracting Features", ncols=80):
            # count += 1
            # if count > 200:
            #     break
            image = self._load_image(image_path)

            if self.mode in ['train', 'val']:
                augmented_images = get_augmented_images(image)
                for augmented_image in augmented_images:
                    inputs = self.processor(images=augmented_image, return_tensors="pt", padding=True)
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}

                    with torch.no_grad():
                        image_embeds = self.model.get_image_features(inputs['pixel_values'])
                        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
                        features.append(image_embeds.cpu().numpy())
                        indices.append(int(image_path.stem))
            else:
                inputs = self.processor(images=image, return_tensors="pt", padding=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                with torch.no_grad():
                    image_embeds = self.model.get_image_features(inputs['pixel_values'])
                    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
                    features.append(image_embeds.cpu().numpy())
                    indices.append(int(image_path.stem))

        return np.vstack(features), indices

def save_model(model, path):
    """Save scikit-learn SVM model using joblib"""
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved successfully to {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(path):
    """Load scikit-learn SVM model using joblib"""
    try:
        clf = joblib.load(path)
        logging.info(f"Model loaded successfully from {path}")
        return clf
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def sfs(train_data, train_labels, val_data, val_labels, gamma, C):
    # Sequential Forward Selection
    best_score = 0
    features = []
    best_features = []
    remaining_features = list(range(train_data.shape[1]))
    while remaining_features:
        best_score_local = 0
        best_feature = None
        for feature in remaining_features:
            clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
            selected_features = features + [feature]
            train_data_new = train_data[:, selected_features]
            clf.fit(train_data_new, train_labels)
            val_pred = clf.predict(val_data[:, selected_features])
            f1_score_macro = f1_score(val_labels, val_pred, average='macro')
            if f1_score_macro > best_score_local:
                best_score_local = f1_score_macro
                best_feature = feature
        features.append(best_feature)
        remaining_features.remove(best_feature)
        if best_score_local > best_score:
            best_score = best_score_local
            best_features = features.copy()

    return best_features, best_score

def _grid_search_helper(params):
    C, gamma, train_data, train_labels, val_data, val_labels = params
    sfs_features, sfs_score = sfs(train_data, train_labels, val_data, val_labels, gamma, C)
    return {'C': C, 'gamma': gamma, 'features': sfs_features, 'score': sfs_score}

def grid_search(train_data, train_labels, val_data, val_labels):
    # Grid search for hyperparameter tuning
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # Create parameter combinations
    params = [(C, gamma, train_data, train_labels, val_data, val_labels) 
             for C, gamma in product(C_values, gamma_values)]
    
    # Run grid search in parallel with progress bar and CPU control
    num_cpus = os.cpu_count() // 2  # Use half of available CPUs by default
    with Pool(processes=num_cpus) as pool:
        results = list(tqdm(
            pool.imap(_grid_search_helper, params),
            total=len(params),
            desc="Grid Search Progress",
            ncols=70
        ))
    
    # Find best parameters
    best_params = max(results, key=lambda x: x['score'])

    # Create and plot heatmap
    scores_matrix = np.zeros((len(C_values), len(gamma_values)))
    for result in results:
        i = C_values.index(result['C'])
        j = gamma_values.index(result['gamma'])
        scores_matrix[i, j] = result['score']

    plt.figure(figsize=(10, 8))
    plt.imshow(scores_matrix, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(label='F1 Score')
    
    # Set axis labels
    plt.xticks(range(len(gamma_values)), [f'{g:.3f}' for g in gamma_values], rotation=45)
    plt.yticks(range(len(C_values)), [f'{c:.3f}' for c in C_values])
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.title('Grid Search Results Heatmap')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('./results/grid_search_heatmap.png')
    plt.close()
    
    return best_params

def train(args):
    if os.path.exists('./results/train_data.npz'):
        logging.info("Found pre-extracted train data. Loading...")
        train_data = np.load('./results/train_data.npz', allow_pickle=True)['features']
        train_labels = np.load('./results/train_data.npz', allow_pickle=True)['labels']
    else:
        train_extractor = ImageFeatureExtractor(original_images_dir=args.train_data, clip_model=args.CLIP_model, mode='train')
        train_data, train_labels = [], []
        for label, category in enumerate(args.classes_list):
            image_paths = train_extractor._get_image_paths(category)
            features, _ = train_extractor.extract_features(image_paths)
            train_data.append(features)
            train_labels.extend([label] * len(features))

        train_data = np.vstack(train_data)
        train_labels = np.array(train_labels)

        # Save extracted features and labels
        np.savez('./results/train_data.npz', features=train_data, labels=train_labels)

    if os.path.exists('./results/val_data.npz'):
        logging.info("Found pre-extracted validation data. Loading...")
        val_data = np.load('./results/val_data.npz', allow_pickle=True)['features']
        val_labels = np.load('./results/val_data.npz', allow_pickle=True)['labels']
    else:
        val_data = []
        val_extractor = ImageFeatureExtractor(original_images_dir=args.val_data, clip_model=args.CLIP_model, mode='val')
        image_paths = val_extractor._get_image_paths()
        features, label_indices = val_extractor.extract_features(image_paths)
        val_data.append(features)
        val_data = np.vstack(val_data)

        labels_df = pd.read_excel((Path(args.val_data) / Path('Validation_labels.xlsx')))
        logging.info(f"Loaded {len(labels_df)} validation labels from {(Path(args.val_data) / Path('Validation_labels.xlsx'))}")

        val_labels = [labels_df.iloc[idx]['Label_B'] for idx in label_indices]

        # Save extracted features
        np.savez('./results/val_data.npz', features=val_data, labels=val_labels)

    logging.info(f"Train data shape: {train_data.shape}, Val data shape: {val_data.shape}")

    if os.path.exists('./results/selected_features.npz'):
        selected_features_indices = np.load('./results/selected_features.npz', allow_pickle=True)['features']
        train_data_new = train_data[:, selected_features_indices]
        val_data_new = val_data[:, selected_features_indices]
        logging.info(f"Selected {len(selected_features_indices)} features for training")
    else:
        selector = SelectKBest(score_func=f_classif, k=50)
        train_data_new = selector.fit_transform(train_data, train_labels)
        selected_features_indices = selector.get_support(indices=True)
        np.savez('./results/selected_features.npz', features=selected_features_indices)
        train_data_new = train_data[:, selected_features_indices]
        val_data_new = val_data[:, selected_features_indices]
    
  
    if os.path.exists('./results/best_features_of_selected_features.npz'):
        best_features = np.load('./results/best_features_of_selected_features.npz', allow_pickle=True)['features']
        train_data_new = train_data_new[:, best_features]
        val_data_new = val_data_new[:, best_features]
        clf = svm.SVC(kernel='rbf', C=10, gamma=1, class_weight='balanced')
        logging.info(f"Selected {len(best_features)} features for training")
    else:
        best_params = grid_search(train_data_new, train_labels, val_data_new, val_labels)
        train_data_new = train_data_new[:, best_params['features']]
        val_data_new = val_data_new[:, best_params['features']]
        clf = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], class_weight='balanced')
        np.savez('./results/best_features_of_selected_features.npz', features=best_params['features'])
        logging.info(f"Best hyperparameters: {best_params}")

    clf.fit(train_data_new, train_labels)

    val_pred = clf.predict(val_data_new)

    balanced_acc = balanced_accuracy_score(val_labels, val_pred)
    f1_score_macro = f1_score(val_labels, val_pred, average='macro')
    logging.info(f"Balanced Accuracy: {balanced_acc:.4f}, F1 Score (macro): {f1_score_macro:.4f}")

    # Save model
    if args.model_output:
        save_model(clf, args.model_output)
        



def validate(args):
    if os.path.exists('./results/val_data.npz'):
        logging.info("Found pre-extracted validation data. Loading...")
        val_data = np.load('./results/val_data.npz', allow_pickle=True)['features']
        val_labels = np.load('./results/val_data.npz', allow_pickle=True)['labels']
    else:
        val_extractor = ImageFeatureExtractor(original_images_dir=args.val_data, clip_model=args.CLIP_model, mode='val')
        val_data = []
        image_paths = val_extractor._get_image_paths()
        features, label_indices = val_extractor.extract_features(image_paths)
        val_data.append(features)
        val_data = np.vstack(val_data)

        labels_df = pd.read_excel((Path(args.val_data) / Path('Validation_labels.xlsx')))
        logging.info(f"Loaded {len(labels_df)} validation labels from {(Path(args.val_data) / Path('Validation_labels.xlsx'))}")
        val_labels = [labels_df.iloc[idx]['Label_B'] for idx in label_indices]

    if os.path.exists('./results/selected_features.npz'):
        selected_features_indices = np.load('./results/selected_features.npz', allow_pickle=True)['features']
        val_data = val_data[:, selected_features_indices]
        best_features = np.load('./results/best_features_of_selected_features.npz', allow_pickle=True)['features']
        val_data = val_data[:, best_features]
        logging.info(f"Selected {len(best_features)} features for validation")

    clf = load_model(args.model_output)
    val_pred = clf.predict(val_data)

    balanced_acc = balanced_accuracy_score(val_labels, val_pred)
    f1_score_macro = f1_score(val_labels, val_pred, average='macro')
    logging.info(f"Balanced Accuracy: {balanced_acc:.4f}, F1 Score (macro): {f1_score_macro:.4f}")

def test(args):
    if os.path.exists('./results/test_data.npz'):
        logging.info("Found pre-extracted test data. Loading...")
        test_data = np.load('./results/test_data.npz', allow_pickle=True)['features']
    else:
        test_extractor = ImageFeatureExtractor(original_images_dir=args.test_data, clip_model=args.CLIP_model, mode='test')
        test_data = []
        image_paths = test_extractor._get_image_paths()
        features, indices = test_extractor.extract_features(image_paths)
        test_data.append(features)
        test_data = np.vstack(test_data)
        # Sort test_data by indices
        test_data = [data for _, data in sorted(zip(indices, test_data))]
        # Save extracted features
        np.savez('./results/test_data.npz', features=test_data)

    if os.path.exists('./results/selected_features.npz'):
        selected_features_indices = np.load('./results/selected_features.npz', allow_pickle=True)['features']
        test_data = test_data[:, selected_features_indices]
        best_features = np.load('./results/best_features_of_selected_features.npz', allow_pickle=True)['features']
        test_data = test_data[:, best_features]
        logging.info(f"Selected {len(best_features)} features for testing")

    clf = load_model(args.model_output)
    test_pred = clf.predict(test_data)

    labels_df = pd.read_excel((Path(args.test_data) / Path('captions.xlsx')))

    task1_predictions = [0 if pred == 0 else 1 for pred in test_pred]
    df = pd.DataFrame({'index': labels_df['Index'], 'caption':labels_df['Caption'],  'Label_A': task1_predictions,'Label_B': test_pred})
    df.to_json('./results/answer.json', orient='records', indent=4)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Image Classification with CLIP features')
    parser.add_argument('--train_data', type=str, default="../data/train", help='Directory containing image categories')
    parser.add_argument('--val_data', type=str, default="../data/val", help='Directory containing validation images')
    parser.add_argument('--test_data', type=str, default="../data/test", help='Directory containing test images')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--val', action='store_true', help='Validate the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--CLIP_model', type=str, default="openai/clip-vit-base-patch16", help='CLIP model to use')
    parser.add_argument('--model_output', type=str, default="./results/svm_model.pth", help='Path to save trained model (optional)')
    parser.add_argument('--classes_list', type=str, nargs='+', default=["coco_image", "sd21_image", "sdxl_image", "sd3_image", "dalle_image", "midjourney_image"], help='list of class names to use (default: "coco_image", "sd3_image", "sd21_image", "sdxl_image", "dalle_image", "midjourney_image")')

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.val:
        validate(args)
    elif args.test:
        test(args)