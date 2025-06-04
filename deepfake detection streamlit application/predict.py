import os
import torch
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
import pretrainedmodels

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === MODELS ===
# Load Xception model
xception_model = pretrainedmodels.__dict__['xception'](pretrained='imagenet').to(device)
xception_model.last_linear = torch.nn.Linear(xception_model.last_linear.in_features, 128).to(device)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Mediapipe for landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# COCO classes
COCO_CLASSES = yolo_model.names

# === DEFINE CLASSIFIERS ===
class DeepfakeClassifier(torch.nn.Module):
    def __init__(self):
        super(DeepfakeClassifier, self).__init__()
        self.xception = xception_model
        self.sobel_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.sobel_linear = torch.nn.Linear(32 * 74 * 74, 128).to(device)
        self.fc_landmarks = torch.nn.Linear(936, 128).to(device)
        self.fc_yolo = torch.nn.Linear(80, 64).to(device)
        self.fc1 = torch.nn.Linear(128 + 128 + 128 + 64, 128).to(device)
        self.fc2 = torch.nn.Linear(128, 2).to(device)

    def forward(self, image, sobel_image, yolo_features, face_landmarks):
        image_features = self.xception(image)
        sobel_features = self.sobel_cnn(sobel_image)
        sobel_features = sobel_features.view(sobel_features.size(0), -1)
        sobel_features = self.sobel_linear(sobel_features)
        yolo_features = torch.relu(self.fc_yolo(yolo_features.float()))
        landmark_features = torch.relu(self.fc_landmarks(face_landmarks.float()))
        combined = torch.cat((image_features, sobel_features, yolo_features, landmark_features), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        return x

class DNN(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim_1=128, hidden_dim_2=256, output_dim=2):
        super(DNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, output_dim)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class EnsembleModel(torch.nn.Module):
    def __init__(self, module1_dim=2, module2_dim=2, output_dim=2):
        super(EnsembleModel, self).__init__()
        self.fc1 = torch.nn.Linear(module1_dim + module2_dim, output_dim)

    def forward(self, x1_logits, x2_logits):
        x1_probs = torch.softmax(x1_logits, dim=1)
        x2_probs = torch.softmax(x2_logits, dim=1)
        combined = torch.cat((x1_probs, x2_probs), dim=1)
        return self.fc1(combined)

# === UTILITY FUNCTIONS ===
def generate_sobel(image):
    gray = cv2.cvtColor(image.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.convertScaleAbs(magnitude)
    sobel = cv2.merge([magnitude, magnitude, magnitude])
    return transform(Image.fromarray(sobel))

def extract_features(image_tensor):
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    results = yolo_model(img_np)[0]
    detected = []
    landmarks = np.zeros((936,), dtype=np.float32)

    for box in results.boxes:
        class_id = int(box.cls[0])
        detected.append(class_id)
        if COCO_CLASSES[class_id] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img_np[y1:y2, x1:x2]
            face_result = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if face_result.multi_face_landmarks:
                landmarks = np.array(
                    [[p.x, p.y] for p in face_result.multi_face_landmarks[0].landmark]
                ).flatten()
    yolo_vec = torch.tensor([1 if i in detected else 0 for i in range(len(COCO_CLASSES))])
    return yolo_vec, torch.tensor(landmarks)



def predict(image_path, csv_path):
    print(f"Processing image: {image_path}")
    print(f"Using CSV file: {csv_path}")
    
    try:
        # Load CSV features
        df = pd.read_csv(csv_path)
        print(f"CSV contents:\n{df}")
        if df.empty:
            raise ValueError("No features found in CSV file")

        # Handle both string paths and PIL Images
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
            base_name = os.path.basename(image_path)
        else:
            img = image_path
            base_name = "temp_image"

        # Transform image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Generate Sobel features
        sobel_tensor = generate_sobel(img_tensor.squeeze(0)).unsqueeze(0).to(device)
        
        # Extract YOLO and landmark features
        yolo_vec, landmarks = extract_features(img_tensor.squeeze(0))
        yolo_vec = yolo_vec.unsqueeze(0).to(device)
        landmarks = landmarks.unsqueeze(0).to(device)

        # Find the matching row in CSV
        matching_rows = df[df['image_name'].str.contains(base_name, na=False)]
        if matching_rows.empty:
            raise ValueError(f"No features found for image: {base_name}")
        
        row = matching_rows.iloc[0]
        
        # Convert CSV features to tensor
        csv_features = torch.tensor(
            [float(x) for x in row['features'].strip('[]').split(',')]
        ).unsqueeze(0).to(device)

        # Load models if not already loaded
        module1 = DeepfakeClassifier().to(device)
        module2 = DNN().to(device)
        ensemble = EnsembleModel().to(device)

        # Load model weights
        try:
            module1.load_state_dict(torch.load('models/best_model_module1.pth', map_location=device))
            module2.load_state_dict(torch.load('models/best_model_module2.pth', map_location=device))
            ensemble.load_state_dict(torch.load('models/best_ensemble_model_WildRF.pth', map_location=device))
        except Exception as e:
            raise ValueError(f"Failed to load model weights: {e}")

        # Set models to evaluation mode
        module1.eval()
        module2.eval()
        ensemble.eval()

        # Make predictions
        with torch.no_grad():
            # Get predictions from both modules
            out1 = module1(img_tensor, sobel_tensor, yolo_vec, landmarks)
            out2 = module2(csv_features)
            
            # Ensemble predictions
            final_out = ensemble(out1, out2)
            
            # Get probabilities and prediction
            probs = torch.softmax(final_out, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

            # Map prediction to label
            prediction = "Fake" if pred_class == 1 else "Real"
            
            print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
            return prediction, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0