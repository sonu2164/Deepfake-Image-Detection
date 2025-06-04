# Deepfake Detection App

This project is a deepfake detection application that utilizes ensemble learning to classify images as real or fake. The application leverages multiple deep learning models to enhance prediction accuracy.

## Project Structure

- `src/models/`: Contains the model definitions and architectures.
  - `ensemble_model.py`: Implements the `EnsembleModel` class that combines outputs from different models.
  - `feature_extraction.py`: Defines the `DeepfakeClassifier` and `DNN` classes for feature extraction and classification.
  
- `src/utils/`: Contains utility functions for image processing and model loading.
  - `image_processing.py`: Functions for generating Sobel edges and extracting features.
  - `model_loader.py`: Functions to load pre-trained models and their weights.

- `src/app.py`: The main entry point for the Streamlit application, handling user interactions and predictions.

- `src/config.py`: Configuration settings, including model paths and image transformation parameters.

- `models/`: Directory containing the trained model weights.

- `tests/`: Directory for unit tests.

- `uploaded_images/`: Directory for storing uploaded images.

- `uploaded_images_csv/`: Directory for storing CSV files with extracted features.

- `requirements.txt`: Lists the dependencies required for the project.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd deepfake-detection-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload an image to check if it's real or fake.

## License

This project is licensed under the MIT License. See the LICENSE file for details.