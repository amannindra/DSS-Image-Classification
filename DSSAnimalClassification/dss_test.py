import torch
import torchvision.models as models
import torch.nn as nn
import os
from PIL import Image
import io
import torchvision.transforms as transforms
import json

def model_fn(model_dir):
    print("Loading model from: ", model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 8)
    model_path = os.path.join(model_dir, 'model.pth')
    print("Loading model from: ", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))  
    print("Model loaded successfully")
    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        
        # Same transforms as validation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transform(image).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)
    return probabilities

def output_fn(prediction, response_content_type):
    print("Outputting prediction: ", prediction)
    class_names = [
            "antelope_duiker", "bird", "blank", "civet_genet", 
            "hog", "leopard", "monkey_prosimian", "rodent"
    ]
    probs = prediction.cpu().numpy()[0]
    best_idx = probs.argmax()
    
    response = {
        "predicted_class": class_names[best_idx],
        "confidence": float(probs[best_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }
    
    return json.dumps(response)