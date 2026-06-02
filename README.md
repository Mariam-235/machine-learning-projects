# Machine Learning & Image Segmentation Project

## Overview

This project contains three machine learning tasks implemented in Python. It demonstrates both supervised and unsupervised learning techniques, including classification, clustering, and image segmentation using real datasets.
---

## Project Structure

- **kdd_classification.py** → Network intrusion classification (Perceptron + K-Means)
- **customer_segmentation.py** → Customer segmentation using K-Means clustering
- **image_segmentation.py** → Image segmentation using K-Means on plant images

### Datasets:
- kdd_dataset.csv → Network intrusion dataset (normal vs anomaly)
- mall_customer.csv → Customer behavior dataset
- Plant_Dataset/ → Plant images for segmentation

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow (PIL)

---

## Project Breakdown

### 1. Network Intrusion Classification (KDD Dataset)
- Train/test split (70/30)
- Perceptron classifier implementation
- K-Means clustering comparison
- Accuracy evaluation
---

### 2. Customer Segmentation (Mall Customers Dataset)
- Data preprocessing (scaling + encoding)
- K-Means clustering (k=2,3,4)
- Business insights:
  - Spending behavior
  - Income levels
  - Customer groups

---

### 3. Image Segmentation
- K-Means clustering on image pixels(RGB values)
-Segmented images into multiple clusters (k = 2, 3, 4, 5)
-Generated visual outputs showing each segmented region
-Highlighted dominant color regions per cluster

---

## How to Run

Step 1: Clone the repository
git clone https://github.com/Mariam-235/machine-learning-projects.git

Step 2: Navigate to the project folder
cd machine-learning-projects

Step 3: Install required dependencies

Make sure Python is installed, then run:

pip install -r requirements.txt

Step 4: Run the projects

You can run each script separately:

python kdd_classification.py

python customer_segmentation.py

python image_segmentation.py


## Author

 Mariam Elshazly
