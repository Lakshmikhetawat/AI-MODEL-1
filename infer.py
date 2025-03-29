import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the CNN Model (Must match the trained model)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (Ensure this matches your dataset)
class_names = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

num_classes = len(class_names)  # Set the correct number of classes

# Load the trained model
model = CNNModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("tomato_disease_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Define image transformations (Must match the training process)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to make a prediction
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    print(f"Predicted Class: {predicted_class}")
    return predicted_class

# Run inference
if __name__ == "__main__":
    image_path = "C:\\Users\\Lakshmi Khetawat\\Desktop\\hackrux1\\plant-detect\\data\\img.jpg"  # Change this to the path of your image
    predict(image_path)