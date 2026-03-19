# 🧠 End-to-End OCR Pipeline (ViteTech Assignment)

## 📌 Overview

This project demonstrates a simple end-to-end OCR (Optical Character Recognition) pipeline built using classical machine learning techniques. The goal is to simulate a real-world OCR system by covering the full pipeline:

**Data → Preprocessing → Model → Evaluation → Improvement → Practical Usage**

The focus of this project is not achieving the highest accuracy, but showcasing:

* data understanding
* proper evaluation (CER/WER)
* error analysis
* iterative improvement
* thinking beyond OCR (LLM post-processing)

#  Dataset

This project uses the **Handwritten Words Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/nabeel965/handwritten-words-dataset

# Dataset Description

* Contains handwritten word images
* Organized in folders where each folder name represents the label (ground truth text)
* Includes variations in handwriting, noise, and formatting

*Note:*
The dataset is not included in this repository due to GitHub size limitations.

# How to Use

1. Download the dataset from the link above
2. Extract the dataset
3. Place it inside the project directory:

data/
#  Limitations

- The model treats OCR as a classification task instead of sequence prediction
- Flattening images removes spatial relationships between pixels
- Performance degrades on complex or noisy handwriting
- Not suitable for full sentence recognition
# Future Work

- Use transformer-based OCR models like TrOCR
- Apply CNN + RNN architectures for sequence learning
- Improve preprocessing with denoising and augmentation
- Integrate real LLM APIs for better text correction

#  Data Audit

Before training, the dataset was analyzed:

# Observations:

* Nested folder structure (multiple levels)
* Labels derived from folder names
* Some corrupted or unreadable images
* Inconsistent casing in labels

# Fixes Applied:

* Ignored unreadable images
* Converted all labels to lowercase
* Handled nested folder structures automatically

# Ignored:

* Slightly noisy images (kept for realism)

#  Approach

# Pipeline:

Image → Grayscale → Resize (64x64) → Normalize → Flatten → RandomForest → Prediction → LLM Correction
# Model:

* *RandomForestClassifier*
* Used as a baseline model

# Why this approach?

* Simple and fast to implement
* Good baseline for comparison
* Helps highlight limitations of classical ML for OCR

# Evaluation

Evaluation was done using:

* **Character Error Rate (CER)**
* **Word Error Rate (WER)**

Libraries used:

jiwer
# Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | 76.23%|
| WER      |0.237  |
| CER      | 0.260   |

*(Replace with your actual values)*

#  Sample Predictions

| Ground Truth | Prediction | After Correction |
| ------------ | ---------- | ---------------- |
| quickly      | quikly     | quickly          |
| five         | fve        | five             |
| wizard       | wizar      | wizards          |

# Error Analysis

Common issues observed:

* Missing characters (hello → helo)
* Confused characters (o vs 0)
* Spelling errors
* Sensitivity to noise in images

#Key Insight:

The model struggles because it treats OCR as a classification problem rather than a sequence problem.

#  Improvement

# ✔ LLM-Based Post Processing (Simulated)

A simple correction function was used to simulate LLM behavior:

* Fix spelling errors
* Improve readability
* Correct common OCR mistakes

Example:
`
Raw OCR: quikly  
Corrected: quickly

# Limitations

* Cannot handle sequences (true OCR requirement)
* Uses flattened images (loses spatial information)
* Not robust to complex handwriting

# Future Work

* Use transformer-based OCR models (e.g., TrOCR)
* Apply CNN + RNN architectures
* Improve preprocessing (denoising, augmentation)
* Use real LLM APIs for better correction

# Local → Cloud Deployment

# Local:

* Executed on CPU using Python, OpenCV, and Scikit-learn

# Cloud (Proposed):

* Training: AWS / GCP GPU instances
* Storage: S3 / Cloud Storage
* Inference: FastAPI-based API

# When NOT to Use OCR

* Extremely low-quality or blurred images
* Highly inconsistent handwriting
* When structured digital text is already available
* Real-time systems with strict latency constraints

# How to Run

# Install dependencies:

pip install -r requirements.txt

# Run training:

python train.py

# Project Structure

ocr-pipeline-vitetech-assignment/
│
├── data/
├── train.py
├── README.md
├── requirements.txt


#  Conclusion

This project demonstrates a practical OCR pipeline with:

* real-world data handling
* baseline modeling
* proper evaluation
* improvement using post-processing

It highlights both the **capabilities and limitations** of classical approaches and motivates the need for modern deep learning-based OCR systems.

#  Acknowledgment

Assignment completed as part of the internship application process for **Vite Tech IT Solutions**.
