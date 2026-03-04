# AI-Based Water Pollution Detection from Images

This project is a complete end-to-end machine learning system that allows users to upload an image of a water body (river, lake, pond, or sea) and automatically classifies whether the water is clean or polluted. It also detects common pollution types such as oil contamination, plastic waste, or algae bloom.

## Project Structure
- `dataset/`: Directory for storing the training and validation images.
- `model/`: Directory where the trained PyTorch model is saved.
- `training/`: Contains scripts for dataset preprocessing, model training, and evaluation.
- `backend/`: FastApi inference module to process the image and make predictions.
- `app.py`: The main FastAPI backend application.
- `frontend/`, `static/`, `templates/`: HTML, CSS, and JS components for the web interface.

## Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
(Optional, if you want to train from scratch using your dataset)
1. Organize your dataset inside `dataset/` with subfolders for each class (`clean`, `oil`, `plastic`, `algae`).
2. Run the training script:
   ```bash
   python training/train.py
   ```
3. Evaluate the model:
   ```bash
   python training/evaluate.py
   ```

### 2. Running the Web Application
1. Ensure the latest trained model is inside the `model/` folder.
2. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
3. Open a web browser and navigate to `http://localhost:8000` to access the web interface.
# AI_Water_Pollution_Detection
