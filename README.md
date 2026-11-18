# Breast Cancer Classification with SVM, K-NN and PCA

A machine-learning study that classifies breast tumours as benign or malignant from diagnostic measurements, and tests whether reducing the feature space with PCA changes performance.

## Overview

The dataset has 569 samples and 30 numerical features describing cell nuclei (radius, texture, perimeter and similar measurements). 357 samples are benign and 212 are malignant. The notebook explores the data, trains and compares classifiers, then repeats the experiments after PCA.

## Methods

1. Data exploration: feature scatter plot, standardisation with StandardScaler, PCA and scree plot
2. Data preparation: stratified 80/20 train/test split (455 training and 114 test samples, random_state=42)
3. Classification on all 30 features:
   - SVM with a linear kernel (C = 2, 10, 50)
   - SVM with an RBF kernel (C = 2, 10, 50 and gamma = 0.01, 0.1, 1)
   - K-NN (K = 3, 5, 7)
4. Classification after PCA, keeping 7 components (about 91% of the variance, chosen from the scree plot)
5. Evaluation with accuracy, precision, recall, F1 score and confusion matrices

## Results

Test-set results for the best configuration of each model:

| Setting | Model | Accuracy | F1 score |
|---|---|---|---|
| All 30 features | SVM, RBF (C=10, gamma=0.01) | 0.9825 | 0.9756 |
| All 30 features | K-NN (K=5) | 0.9561 | 0.9383 |
| 7 PCA components | SVM, RBF (C=50, gamma=0.01) | 0.9825 | 0.9756 |
| 7 PCA components | K-NN (K=5) | 0.9561 | 0.9383 |

Key findings:
- SVM outperformed K-NN in both settings.
- Reducing 30 features to 7 principal components left the best accuracy and F1 unchanged, which suggests many of the original features are redundant.
- RBF models with gamma = 1 performed poorly (the model overfits), which shows how sensitive SVMs are to hyperparameters.

## Repository Structure

```
.
├── breast_cancer_classification.ipynb
├── requirements.txt
└── README.md
```

## Running the Notebook

```bash
git clone https://github.com/haneen09/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
jupyter notebook breast_cancer_classification.ipynb
```

The notebook downloads the dataset automatically with `kagglehub`, so an internet connection is required.

## Dataset

[Diagnostic Breast Cancer Dataset](https://www.kaggle.com/datasets/ahmeduzaki/diagnostic-breast-cancer-dataset) on Kaggle. It is not redistributed in this repository.

## Limitations and Future Work

- Results come from a single train/test split on a small dataset, so cross-validation would give more reliable estimates.
- In the PCA experiments, the scaler and PCA were fitted on the full dataset before splitting, which can leak information from the test set. Fitting them on the training set only would be more rigorous.
- Hyperparameters were tested on a small grid rather than tuned systematically.
- This is an educational project and is not intended for clinical use.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, kagglehub

## Author

Your name - [GitHub](https://github.com/haneen09)
