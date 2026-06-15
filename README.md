![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy&logoColor=white) 
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-success) 


# Machine Learning Algorithms Implementation

A comprehensive collection of machine learning models implemented using Python and scikit-learn, covering various supervised learning algorithms for both classification and regression tasks.

## Overview

This repository contains **7 Jupyter notebooks** demonstrating the implementation of fundamental machine learning algorithms on real-world datasets. Each notebook covers data preprocessing, model training, evaluation, and visualization techniques.

## Project Structure

```text
├── NaiveBayes_KeshavVerma.ipynb          # Naive Bayes classifier for spam detection
├── DecisionTree_KeshavVerma.ipynb        # Decision Tree classifier for drug prediction
├── EnsembleBagging_KeshavVerma.ipynb     # Bagging & Boosting ensemble methods
├── RandomForest_KeshavVerma.ipynb        # Random Forest for drug prediction
├── LogisticRegression_KeshavVerma.ipynb  # Logistic Regression for diabetes prediction
├── KNN_KeshavVerma.ipynb                 # K-Nearest Neighbors for tumor classification
└── LinearRegression_KeshavVerma.ipynb    # Linear Regression for car price prediction
```

## Models & Datasets

| Notebook           | Algorithm                | Task           | Dataset                         |
| ------------------ | ------------------------ | -------------- | ------------------------------- |
| NaiveBayes         | Multinomial Naive Bayes  | Classification | Email Spam Dataset              |
| DecisionTree       | Decision Tree Classifier | Classification | Drug Dataset                    |
| EnsembleBagging    | Bagging & AdaBoost       | Classification | Email Spam Dataset              |
| RandomForest       | Random Forest Classifier | Classification | Drug Dataset                    |
| LogisticRegression | Logistic Regression      | Classification | Diabetes Dataset                |
| KNN                | K-Nearest Neighbors      | Classification | Breast Cancer Wisconsin Dataset |
| LinearRegression   | Linear Regression        | Regression     | Car Prices Dataset              |

---

## Dataset Information

### Email Spam Dataset

* **Source:** Kaggle
* **Size:** 5,572 entries, 2 columns (`label`, `message`)
* **Task:** Binary Classification (Spam / Ham)

### Drug Dataset

* **Source:** Kaggle
* **Size:** 200 entries
* **Features:** Age, Sex, BP, Cholesterol, Na_to_K Ratio
* **Task:** Multi-Class Classification (5 Drug Types)

### Diabetes Dataset

* **Source:** Kaggle
* **Size:** 768 entries
* **Features:**

  * Pregnancies
  * Glucose
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * BMI
  * Diabetes Pedigree Function
  * Age
* **Task:** Binary Classification (Diabetic / Non-Diabetic)

### Breast Cancer Wisconsin Dataset

* **Source:** Kaggle
* **Size:** 569 entries, 33 features
* **Task:** Binary Classification (Malignant / Benign)

### Car Prices Dataset

* **Source:** Kaggle
* **Size:** 301 entries, 9 features
* **Task:** Regression (Selling Price Prediction)

---

## Technologies & Libraries

### Core Technologies

* Python 3.x
* Jupyter Notebook

### Libraries

* **pandas** – Data manipulation and analysis
* **numpy** – Numerical computing
* **matplotlib** – Data visualization
* **scikit-learn** – Machine learning algorithms and utilities

### scikit-learn Modules Used

```python
sklearn.model_selection
sklearn.preprocessing
sklearn.feature_extraction.text
sklearn.metrics
sklearn.ensemble
sklearn.tree
```

---

## Model Performance Summary

| Model               | Accuracy | Dataset               |
| ------------------- | -------- | --------------------- |
| Naive Bayes         | 98.39%   | Email Spam            |
| Decision Tree       | 100.00%  | Drug Dataset          |
| Bagging Ensemble    | 97.22%   | Email Spam            |
| AdaBoost            | 90.58%   | Email Spam            |
| Random Forest       | 100.00%  | Drug Dataset          |
| Logistic Regression | 82.47%   | Diabetes Dataset      |
| K-Nearest Neighbors | 94.74%   | Breast Cancer Dataset |

> **Note:** The Drug Dataset contains only 200 samples. The 100% accuracy achieved by Decision Tree and Random Forest models is largely due to the small dataset size and strong feature correlations.

---

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Notebooks

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Open any notebook in:

   * Jupyter Notebook
   * JupyterLab
   * Google Colab

3. Run all cells sequentially.

### Google Colab Dataset Loading

The notebooks are configured to load datasets directly from Google Drive:

```python
import pandas as pd

file_id = "YOUR_FILE_ID"
file_url = f"https://drive.google.com/uc?id={file_id}"

df = pd.read_csv(file_url)
```

---

## Key Implementation Details

### Data Preprocessing

* Label Encoding for categorical variables
* One-Hot Encoding where appropriate
* Feature scaling using `StandardScaler`
* Text vectorization using:

  * `CountVectorizer`
  * `TF-IDF Vectorizer`
* Handling missing values
* Removing irrelevant features

### Model Evaluation Metrics

* Confusion Matrix
* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report
* Cross-Validation (Logistic Regression)

### Visualizations

* Decision Tree visualization
* Feature Importance plots
* Confusion Matrices
* Performance comparison charts

---

## Learning Outcomes

This project demonstrates:

1. End-to-end machine learning workflows
2. Data preprocessing and feature engineering
3. Binary and multi-class classification techniques
4. Regression modeling
5. Ensemble learning methods (Bagging & Boosting)
6. Text classification using TF-IDF
7. Model evaluation and performance analysis
8. Visualization of machine learning results

---

## Notes

* All datasets are loaded from publicly accessible Google Drive links.
* Random states are fixed to ensure reproducibility.
* The Drug Dataset is intentionally small and suitable for educational purposes.
* The Email Spam Dataset uses TF-IDF vectorization for feature extraction.
* The notebooks are designed for learning and experimentation with machine learning algorithms.

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* ROC-AUC analysis and visualization
* More advanced evaluation metrics
* Cross-validation for all models
* Class imbalance handling techniques
* Feature selection and dimensionality reduction
* Model deployment examples

---

## Author

**Keshav Verma**

Course Period
3rd Semester | Uploaded: June 15, 2026

---

### If you found this project helpful, consider giving it a star!

*This project was created for educational purposes to demonstrate fundamental machine learning concepts and practical implementations using Python and scikit-learn.*
