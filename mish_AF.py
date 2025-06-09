import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import skimage.measure
import skimage.morphology as morph
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset path
dataset_path = "C:/Users/Zaid Chikte/Downloads/Diagnosis of Diabetic Retinopathy.v3i.folder"  # Update based on your folder structure
image_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))
print(f"Total images found: {len(image_paths)}")

# Define Mish activation function
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Function for feature extraction
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    
    # Extract Retinal Vessel Diameter (AVR) - Placeholder
    avr = np.random.uniform(0.5, 0.9)
    
    # Extract Vessel Tortuosity - Placeholder
    tortuosity = np.random.uniform(100, 600)
    
    # Extract Microaneurysms & Hemorrhages - Placeholder
    microaneurysms = np.random.randint(0, 5000)
    
    # Extract Optic Disc & Cup-to-Disc Ratio (CDR) - Placeholder
    cdr = np.random.uniform(0.3, 0.8)
    
    # Extract Fractal Dimension - Placeholder
    fractal_dimension = np.random.uniform(5.0, 7.0)
    
    return [avr, tortuosity, microaneurysms, cdr, fractal_dimension]

# Generate feature dataset
features = [extract_features(img_path) for img_path in image_paths]

# Risk classification function
def classify_risk(avr, tortuosity, microaneurysms, cdr, fractal_dimension):
    risk_score = 0
    if avr < 0.6: risk_score += 2
    elif 0.6 <= avr < 0.7: risk_score += 1
    if tortuosity > 500: risk_score += 2
    elif 300 <= tortuosity <= 500: risk_score += 1
    if microaneurysms > 2000: risk_score += 2
    elif 500 <= microaneurysms <= 2000: risk_score += 1
    if cdr > 0.6: risk_score += 2
    elif 0.4 <= cdr <= 0.6: risk_score += 1
    if fractal_dimension < 5.5: risk_score += 2
    elif 5.5 <= fractal_dimension <= 6.5: risk_score += 1
    return "Low Risk" if risk_score <= 2 else "Medium Risk" if risk_score <= 5 else "High Risk"

# Assign risk labels based on extracted features
labels = [classify_risk(*feature) for feature in features]

# Convert labels to numerical format
def encode_labels(label):
    return {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}[label]

labels = [encode_labels(label) for label in labels]

# Create DataFrame
df = pd.DataFrame(features, columns=["AVR", "Tortuosity", "Microaneurysms", "CDR", "Fractal_Dimension"])
df["Image_Path"] = image_paths
df["Risk_Label"] = labels

# Save extracted features and labels to CSV
csv_path = "C:/Users/Zaid Chikte/Desktop/ResNet-LSTM (new dataset)/retinal_features_1.csv"
df.to_csv(csv_path, index=False)
print(f"Feature extraction completed. CSV saved at: {csv_path}")

# Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Risk_Label"], random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Check for dataset imbalance and compute class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define Dataset Class
class RetinalFundusDataset(Dataset):
    def __init__(self, image_paths, transform=None, labels=None):
        self.image_paths = image_paths
        self.transform = transform
        self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            if self.labels is not None:
                label = self.labels[idx]
                return image, torch.tensor(label, dtype=torch.long)
            else:
                return image
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None, None

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Convert DataFrame into PyTorch Dataset
train_dataset = RetinalFundusDataset(train_df["Image_Path"].tolist(), train_transform, train_df["Risk_Label"].tolist())
test_dataset = RetinalFundusDataset(test_df["Image_Path"].tolist(), transform, test_df["Risk_Label"].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Hybrid ResNet-LSTM Model
class HybridResNetLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridResNetLSTM, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        # Freeze ResNet layers to use it as a feature extractor
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer
        
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(512, 256)
        self.activation = Mish()  # Use Mish activation
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.resnet(x)  # Extract features from ResNet50
        x = x.unsqueeze(1)  # Adding sequence length dimension for LSTM
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])  
        x = self.activation(x)  # Apply Mish activation
        x = self.fc2(x)  
        return x


# Model Training
model = HybridResNetLSTM().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


import matplotlib.pyplot as plt

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(10):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if images is None or labels is None:
            continue

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        scheduler.step()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

        print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct_preds / total_samples
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    epoch_val_loss = val_running_loss / len(test_loader)
    epoch_val_acc = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch {epoch+1} Summary:")
    print(f"  → Training Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.4f}")
    print(f"  → Validation Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.4f}\n")

# Plotting training and validation metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.tight_layout()
plt.show()


torch.save(model.state_dict(), "hybrid_resnet_lstm.pth")


import os
import glob 
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# ======= Setup =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = HybridResNetLSTM().to(device)
model.load_state_dict(torch.load("hybrid_resnet_lstm.pth", map_location=device))
model.eval()

# Image pre-processing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Label decoder
label_decoder = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# ======= Prediction Function =======
def predict_risks(input_folder, output_csv_path):
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    results = []

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_label = torch.argmax(output, dim=1).item()
                risk = label_decoder[predicted_label]

            print(f"{os.path.basename(path)} → Predicted Risk: {risk}")
            results.append((os.path.basename(path), risk))

        except Exception as e:
            print(f"Failed to process {path}: {e}")

    # Save to CSV
    df = pd.DataFrame(results, columns=["Image_Name", "Predicted_Risk"])
    df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to {output_csv_path}")

# Example usage
predict_risks(
    input_folder="C:/Users/Zaid Chikte/Downloads/Detection Of Diabetic Retinopathy Using Machine Learning.v1-retinopath-dataset-1.folder",
    output_csv_path="C:/Users/Zaid Chikte/Desktop/ResNet-LSTM (new dataset)/Prediction/Output_Results.csv"
)