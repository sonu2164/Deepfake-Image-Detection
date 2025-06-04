
from pathlib import PosixPath
# image_path = PosixPath("e4e")  # Standard quotes
# train_dir = image_path / "train"  # Standard quotes
# test_dir = image_path / "val"  # Standard quotes

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import ssl
import certifi
import torchvision
import torch.nn.functional as F

ssl._create_default_https_context = ssl.create_default_context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_low_features = []
global_mid_features = []
global_high_features = []
# Data transformations (ResNet-style transformations)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset class
# Custom dataset class

class CustomDatasetNew(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # Get valid image files
        for file_name in os.listdir(root_dir):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_files.append(os.path.join(root_dir, file_name))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.image_files)} images")
        
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path
# Load ResNet model and capture features

def load_saved_resnet_model(model_path, low_level_features, mid_level_features, high_level_features):
    try:
        # Create ResNet model properly
        model = torchvision.models.resnet50(pretrained=True)
        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Register hooks using global lists
        model.layer1[0].register_forward_hook(
            lambda m, i, o: hook_fn(m, i, o, global_low_features)
        )
        model.layer3[0].register_forward_hook(
            lambda m, i, o: hook_fn(m, i, o, global_mid_features)
        )
        model.layer4[0].register_forward_hook(
            lambda m, i, o: hook_fn(m, i, o, global_high_features)
        )
        
        # Load weights if available
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                print("Loaded model weights successfully")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
        
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        return model
        
    except Exception as e:
        print(f"Error in model loading: {e}")
        raise

# Hook functions to capture ResNet features
# low_level_features, mid_level_features, high_level_features = [], [], []
def hook_fn(module, input, output, storage_list):
    storage_list.clear()  # Clear previous features
    storage_list.append(output.clone().detach())  # Store new features
    print(f"Hook called - Feature shape: {output.shape}")

# Define linear layers to convert ResNet features to 768 dimensions
# Define linear layers to convert ResNet features to 768 dimensions
low_to_768 = nn.Linear(256, 768).to(device)   # For low-level features
mid_to_768 = nn.Linear(1024, 768).to(device)  # For mid-level features
high_to_768 = nn.Linear(2048, 768).to(device) # For high-level features

def extract_resnet_features(model, image, low_level_features, mid_level_features, high_level_features):
    # Clear global features
    global_low_features.clear()
    global_mid_features.clear()
    global_high_features.clear()
    
    # Ensure image is on the correct device and has batch dimension
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension if needed
    image = image.to(device)
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(image)  # Forward pass triggers the hooks
        
        # Verify features were collected using global lists
        if not all([global_low_features, global_mid_features, global_high_features]):
            print(f"Feature lengths - Low: {len(global_low_features)}, Mid: {len(global_mid_features)}, High: {len(global_high_features)}")
            raise ValueError("Features were not collected properly by hooks")
            
        # Pool and transform features
        try:
            low_pooled = F.adaptive_avg_pool2d(global_low_features[0], (1, 1)).squeeze()
            mid_pooled = F.adaptive_avg_pool2d(global_mid_features[0], (1, 1)).squeeze()
            high_pooled = F.adaptive_avg_pool2d(global_high_features[0], (1, 1)).squeeze()

            # Handle case where squeeze removed too many dimensions
            if len(low_pooled.shape) == 1:
                low_pooled = low_pooled.unsqueeze(0)
            if len(mid_pooled.shape) == 1:
                mid_pooled = mid_pooled.unsqueeze(0)
            if len(high_pooled.shape) == 1:
                high_pooled = high_pooled.unsqueeze(0)

            # Transform to 768 dimensions
            low_768 = low_to_768(low_pooled)
            mid_768 = mid_to_768(mid_pooled)
            high_768 = high_to_768(high_pooled)

            return low_768, mid_768, high_768
            
        except Exception as e:
            print(f"Error in feature processing: {e}")
            raise

# Function to preprocess the image using ViT's transforms
def pipeline_preprocessor():
    vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    return vit_weights.transforms()

# Function to extract ViT embeddings
def get_vit_embedding(vit_model, image_path):
    preprocessing = pipeline_preprocessor()  # Preprocessing from ViT
    img = Image.open(image_path).convert("RGB")  # Ensure we load image by path (string)
    img = preprocessing(img).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        feats = vit_model._process_input(img)
        batch_class_token = vit_model.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = vit_model.encoder(feats)
        vit_hidden = feats[:, 0]  # CLS token
    return vit_hidden

# Load ViT model
def load_vit_model(pretrained_weights_path):
    try:
        vit_model = torchvision.models.vit_b_16(pretrained=True)
        if os.path.exists(pretrained_weights_path):
            try:
                pretrained_vit_weights = torch.load(pretrained_weights_path, map_location=device)
                vit_model.load_state_dict(pretrained_vit_weights, strict=False)
            except Exception as e:
                print(f"Warning: Could not load ViT weights from {pretrained_weights_path}. Using pretrained model. Error: {e}")
        
        vit_model = vit_model.to(device)
        vit_model.eval()
        return vit_model
    except Exception as e:
        raise Exception(f"Error loading ViT model: {str(e)}")

# Add a sequence dimension (if missing) before applying attention
def ensure_correct_shape(tensor):
    if len(tensor.shape) == 2:  # If shape is [batch_size, embedding_dim]
        tensor = tensor.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, embedding_dim]
    elif len(tensor.shape) == 1:  # If shape is [embedding_dim]
        tensor = tensor.unsqueeze(0).unsqueeze(1)  # Add batch and sequence dimensions: [1, 1, embedding_dim]
    return tensor


# Scaled dot product attention function
def scaled_dot_product_attention(Q, K, V):
    # Ensure Q, K, and V have the correct shapes
    Q = ensure_correct_shape(Q)  # Should be [batch_size, 1, embedding_dim]
    K = ensure_correct_shape(K)  # Should be [batch_size, 1, embedding_dim]
    V = ensure_correct_shape(V)  # Should be [batch_size, 1, embedding_dim]

#     print(f"Q shape after unsqueeze: {Q.shape}, K shape after unsqueeze: {K.shape}, V shape after unsqueeze: {V.shape}")  # Debugging
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32).to(Q.device))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

# Save features for each dataset (train/val/test)
import csv

# Save features for each dataset (train/val/test) as CSV


def save_features_to_csv(model, vit_model, dataset, save_path, low_level_features, mid_level_features, high_level_features):
    print(f"Initial feature lists - Low: {len(low_level_features)}, Mid: {len(mid_level_features)}, High: {len(high_level_features)}")
    

    if len(dataset) == 0:
        print("No images found in dataset")
        return False
    
    print("yaha se")
    


    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # Get the first image
        image, img_path = dataset[0]
        print(f"Image loaded, shape: {image.shape}")
        
        with torch.no_grad():
            low_768, mid_768, high_768 = extract_resnet_features(
                model, image, low_level_features, mid_level_features, high_level_features
            )
            print(f"Features extracted - Low: {low_768.shape}, Mid: {mid_768.shape}, High: {high_768.shape}")
            
            print("yaha bhi")
                # Extract ViT features
            vit_hidden = get_vit_embedding(vit_model, img_path)

                # Apply attention mechanism
            output_1 = scaled_dot_product_attention(vit_hidden, low_768, low_768)
            output_2 = scaled_dot_product_attention(output_1, mid_768, mid_768)
            final_output = scaled_dot_product_attention(output_2, high_768, high_768)

                # Convert features to list
            features = final_output.detach().cpu().numpy().flatten().tolist()

                # Write to CSV
            with open(save_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["image_name", "features", "label"])
                    writer.writerow([os.path.basename(img_path), features, 0])
                
            return True    

            

    except Exception as e:
        print(f"Error accessing dataset: {e}")
        return False