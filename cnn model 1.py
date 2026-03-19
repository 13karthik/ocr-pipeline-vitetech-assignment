import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from jiwer import wer, cer


# CONFIG

DATA_PATH = r"C:\Users\karth\Desktop\data"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5

#  LOAD DATA

images = []
labels = []

for level1 in os.listdir(DATA_PATH):
    level1_path = os.path.join(DATA_PATH, level1)

    if not os.path.isdir(level1_path):
        continue

    for level2 in os.listdir(level1_path):
        level2_path = os.path.join(level1_path, level2)

        if not os.path.isdir(level2_path):
            continue

        for root, _, files in os.walk(level2_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0

                    images.append(img)
                    labels.append(level2.lower())

print("✅ Total images:", len(images))

#  LABEL ENCODING

unique_labels = sorted(list(set(labels)))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
idx_to_label = {i: label for label, i in label_to_idx.items()}

y = np.array([label_to_idx[label] for label in labels])
X = np.array(images)

#  TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  DATASET CLASS

class OCRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(OCRDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(OCRDataset(X_test, y_test), batch_size=BATCH_SIZE)

#  CNN MODEL

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNNModel(len(unique_labels))


#  TRAINING

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")


#  EVALUATION

model.eval()
correct = 0
total = 0

y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        y_pred.extend(predicted.numpy())
        y_true.extend(y_batch.numpy())

        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f" Accuracy: {accuracy*100:.2f}%")


#  LABEL CONVERSION

y_pred_labels = [idx_to_label[i] for i in y_pred]
y_true_labels = [idx_to_label[i] for i in y_true]

#  METRICS

print(" WER:", wer(y_true_labels, y_pred_labels))
print(" CER:", cer(y_true_labels, y_pred_labels))


#  SAMPLE OUTPUT

print("\n Sample Predictions:")
for i in range(5):
    print("GT:", y_true_labels[i])
    print("Pred:", y_pred_labels[i])
    print("------")