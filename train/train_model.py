# train_model.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_path = "data/images"
model_save_path = "model/emotion_model.pt"
class_names_path = "model/class_names.txt"
os.makedirs("model", exist_ok=True)

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Dataset
dataset = datasets.ImageFolder(data_path, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Save class names
with open(class_names_path, "w") as f:
    for cls in class_names:
        f.write(cls + "\n")

# Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_dl):.4f}")

# Save model
torch.save(model.state_dict(), model_save_path)
print("✅ Model saved.")
