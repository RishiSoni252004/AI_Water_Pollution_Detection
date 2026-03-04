import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

model = None
class_names = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path='model/water_pollution_model.pth', classes_path='model/classes.txt'):
    global model, class_names
    
    # Load class names
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:

        class_names = ['algae', 'clean', 'plastic']

    num_classes = len(class_names)

    # Initialize model architecture
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load weights if available
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Predictions will be random untrained outputs.")

    model = model.to(device)
    model.eval()

def predict_image(image_bytes):
    global model, class_names

    if model is None:
        load_model()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence_score:.2f}%",
        "all_probabilities": {class_names[i]: f"{prob.item()*100:.2f}%" for i, prob in enumerate(probabilities)}
    }
