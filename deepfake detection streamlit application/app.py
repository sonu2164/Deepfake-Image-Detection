# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms
# from predict import predict
# from feature_extraction import save_features_to_csv, load_saved_resnet_model, load_vit_model, CustomDatasetNew
# import os

# # Configuration
# CONFIG = {
#     'UPLOAD_DIR': "uploaded_images",
#     'CSV_DIR': "uploaded_images_csv",
#     'RESNET_MODEL_PATH': 'models/pretrained_resnet50_state_dict.pth',
#     'VIT_MODEL_PATH': 'models/pretrained_vit_state_dict.pth'
# }

# # Create directories
# os.makedirs(CONFIG['UPLOAD_DIR'], exist_ok=True)
# os.makedirs(CONFIG['CSV_DIR'], exist_ok=True)

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# @st.cache_resource  # Cache the model loading
# def load_models():
#     try:
#         # Load ResNet model
#         low_level_features, mid_level_features, high_level_features = [], [], []
#         resnet_model = load_saved_resnet_model(
#             CONFIG['RESNET_MODEL_PATH'],
#             low_level_features,
#             mid_level_features,
#             high_level_features
#         )
#         # Load ViT model
#         vit_model = load_vit_model(CONFIG['VIT_MODEL_PATH'])
#         return resnet_model, vit_model
#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         return None, None



# # Image transformations
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])



# def main():
#     st.title("Deepfake Detection App")
#     st.write("Upload an image to check if it's real or fake.")

#     # Load models with progress indicator
#     with st.spinner('Loading models...'):
#         resnet_model, vit_model = load_models()
#         if resnet_model is None or vit_model is None:
#             st.error("Failed to load models. Please check the model files and try again.")
#             return

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#         try:
#             # Load and display image
#             image = Image.open(uploaded_file).convert("RGB")
#             st.image(image, caption="Uploaded Image", use_container_width=True)
            
#             # Save image temporarily
#             temp_path = os.path.join(CONFIG['UPLOAD_DIR'], uploaded_file.name)
#             image.save(temp_path)
            
#             with st.spinner('Processing image...'):
#                 # Create dataset with single image
#                 test_dataset = CustomDatasetNew(
#                     root_dir=CONFIG['UPLOAD_DIR'], 
#                     transform=data_transforms
#                 )
                
#                 if len(test_dataset) == 0:
#                     st.error("Failed to process the uploaded image.")
#                     return
                
#                 # Extract and save features
#                 csv_path = os.path.join(CONFIG['CSV_DIR'], 'test_features.csv')
#                 if not save_features_to_csv(resnet_model, vit_model, test_dataset, csv_path):
#                     st.error("Failed to extract features from the image.")
#                     return
                
#                 # Make prediction
#                 result, prob = predict(temp_path, csv_path)
                
#                 if result == "Error":
#                     st.error("Failed to make prediction.")
#                     return
                
#                 # Display results
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Prediction", result)
#                 with col2:
#                     st.metric("Confidence", f"{prob*100:.2f}%")
            
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#         finally:
#             # Cleanup
#             if 'temp_path' in locals():
#                 try:
#                     os.remove(temp_path)
#                 except:
#                     pass
#             if 'csv_path' in locals():
#                 try:
#                     os.remove(csv_path)
#                 except:
#                     pass

# if __name__ == "__main__":
#     main()


import streamlit as st
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from predict import predict
from feature_extraction import save_features_to_csv, load_saved_resnet_model, load_vit_model, CustomDatasetNew
import os
import sys
import logging
# Add before main()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Version checks
REQUIRED_VERSIONS = {
    'torch': '2.1.0',
    'torchvision': '0.16.0',
    'PIL': '10.0.0'
}

def check_versions():
    try:
        if torch.__version__.split('+')[0] != REQUIRED_VERSIONS['torch']:
            st.warning(f"torch version mismatch. Required: {REQUIRED_VERSIONS['torch']}")
        if torchvision.__version__.split('+')[0] != REQUIRED_VERSIONS['torchvision']:
            st.warning(f"torchvision version mismatch. Required: {REQUIRED_VERSIONS['torchvision']}")
        if Image.__version__ != REQUIRED_VERSIONS['PIL']:
            st.warning(f"PIL version mismatch. Required: {REQUIRED_VERSIONS['PIL']}")
    except Exception as e:
        st.error(f"Error checking versions: {str(e)}")

# Configuration
CONFIG = {
    'UPLOAD_DIR': "uploaded_images",
    'CSV_DIR': "uploaded_images_csv",
    'RESNET_MODEL_PATH': 'models/pretrained_resnet50_state_dict.pth',
    'VIT_MODEL_PATH': 'models/pretrained_vit_state_dict.pth'
}

# Create directories
for dir_path in [CONFIG['UPLOAD_DIR'], CONFIG['CSV_DIR']]:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        st.error(f"Error creating directory {dir_path}: {str(e)}")
        sys.exit(1)

# Device configuration with error handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
low_level_features, mid_level_features, high_level_features = [], [], []
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        
        resnet_model = load_saved_resnet_model(
            CONFIG['RESNET_MODEL_PATH'],
            low_level_features,
            mid_level_features,
            high_level_features
        )
        vit_model = load_vit_model(CONFIG['VIT_MODEL_PATH'])
        
        if not all([resnet_model, vit_model]):
            raise ValueError("Failed to load one or more models")
            
        return resnet_model, vit_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    st.set_page_config(
        page_title="Deepfake Detection App",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Deepfake Detection App")
    st.write("Upload an image to check if it's real or fake.")
    
    # Check versions
    check_versions()
    
    # Load models with progress indicator
    with st.spinner('Loading models...'):
        resnet_model, vit_model = load_models()
        if resnet_model is None or vit_model is None:
            st.error("Failed to load models. Please check the model files and try again.")
            return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=300)
            
            # Save image temporarily
            temp_path = os.path.join(CONFIG['UPLOAD_DIR'], uploaded_file.name)
            image.save(temp_path)
            
            progress_text = "Processing image..."
            with st.spinner(progress_text):
                logger.debug(f"Creating dataset from: {CONFIG['UPLOAD_DIR']}")
                test_dataset = CustomDatasetNew(
                    root_dir=CONFIG['UPLOAD_DIR'], 
                    transform=data_transforms
                )
                
                logger.debug(f"Dataset size: {len(test_dataset)}")
                if len(test_dataset) == 0:
                    raise ValueError("Failed to process the uploaded image")
                
                csv_path = os.path.join(CONFIG['CSV_DIR'], 'test_features.csv')
                
                logger.debug("Extracting features...")
                
                if not save_features_to_csv(resnet_model, vit_model, test_dataset, csv_path, low_level_features, mid_level_features, high_level_features):
                    raise ValueError("Failed to extract features from the image")
                
                logger.debug("Making prediction...")
                result, prob = predict(temp_path, csv_path)
                
                if result == "Error":
                    raise ValueError("Failed to make prediction")
                
                # Display results with improved styling
                st.success("Analysis complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", result, delta=None)
                with col2:
                    st.metric("Confidence", f"{prob*100:.2f}%", delta=None)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Cleanup
            for path in [temp_path, csv_path]:
                if 'path' in locals() and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        st.warning(f"Failed to clean up temporary file: {str(e)}")

if __name__ == "__main__":
    main()