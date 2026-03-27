import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#  DATASET PATH

DATA_PATH = r"C:\Users\karth\Desktop\data"

IMG_WIDTH = 64
IMG_HEIGHT = 64

#  PREPROCESS IMAGE

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img.flatten()

#  LOAD DATA (handles nested folders)

images = []
labels = []

for level1 in os.listdir(DATA_PATH):  # Capital / small
    level1_path = os.path.join(DATA_PATH, level1)

    if not os.path.isdir(level1_path):
        continue

    for level2 in os.listdir(level1_path):  # words
        level2_path = os.path.join(level1_path, level2)

        if not os.path.isdir(level2_path):
            continue

        # Case 1: images directly inside
        for file in os.listdir(level2_path):
            img_path = os.path.join(level2_path, file)

            if os.path.isfile(img_path) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = preprocess_image(img_path)

                if img is not None:
                    images.append(img)
                    labels.append(level2.lower())

        # Case 2: nested folder
        for sub in os.listdir(level2_path):
            sub_path = os.path.join(level2_path, sub)

            if not os.path.isdir(sub_path):
                continue

            for file in os.listdir(sub_path):
                img_path = os.path.join(sub_path, file)

                if os.path.isfile(img_path) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = preprocess_image(img_path)

                    if img is not None:
                        images.append(img)
                        labels.append(level2.lower())

#  CHECK DATA

X = np.array(images)

if len(X) == 0:
    print("❌ No images loaded. Check dataset.")
    exit()

print("✅ Total images loaded:", len(X))

#  LABEL ENCODING

unique_labels = sorted(list(set(labels)))
label_to_num = {label: i for i, label in enumerate(unique_labels)}
num_to_label = {i: label for label, i in label_to_num.items()}

y = np.array([label_to_num[label] for label in labels])

print(" Label mapping:", label_to_num)

#  TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  MODEL TRAINING

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

#  EVALUATION

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f" Accuracy: {acc*100:.2f}%")

#  PREDICTION

sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)[0]

raw_output = num_to_label[pred]
actual_output = num_to_label[y_test[0]]

print(" Predicted:", raw_output)
print(" Actual:", actual_output)

#  LLM POST-PROCESSING (SIMULATION)

def llm_correct(text):
    corrections = {
        "wizrds": "wizards",
        "wizar": "wizards",
        "fve": "five",
        "jum": "jump",
        "quikly": "quickly",
        "teh": "the"
    }
    
    return corrections.get(text, text)

corrected_output = llm_correct(raw_output)

print(" Raw OCR Output:", raw_output)
print(" After LLM Correction:", corrected_output)

#  CER & WER EVALUATION

from jiwer import wer, cer

y_pred_labels = [num_to_label[p] for p in y_pred]
y_true_labels = [num_to_label[t] for t in y_test]

print(" WER:", wer(y_true_labels, y_pred_labels))
print(" CER:", cer(y_true_labels, y_pred_labels))
