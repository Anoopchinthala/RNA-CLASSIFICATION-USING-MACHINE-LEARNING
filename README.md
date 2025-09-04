# RNA Classification Using Machine Learning

This mini project explores **RNA classification** using multiple machine learning algorithms.  
The aim is to classify RNA sequences (PCT/mRNA vs LNC) based on their sequence patterns and evaluate performance across different classifiers.

## Features
- Preprocessing of RNA sequences into numerical features
- Implementation of multiple ML models:
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - XGBoost
- Performance metrics:
  - Accuracy, MSE, MAE, R², F1 Score
  - Confusion Matrix
  - ROC Curve
- Comparative plots of all models

## Requirements
- Install dependencies using:
    ```bash
    pip install -r requirements.txt

## Usage
1. Place data_train.tsv and data_test.tsv in the project root.
2. Run the script:
    ```bash
    python Main.py

3. The program will:
- Train ML models
- Print evaluation metrics
- Generate confusion matrices & ROC curves
- Display comparison plots

## Output
### Performance Metrics
| Model               | Accuracy | F1 Score | R² Score | MSE   | MAE   |
| ------------------- | -------- | -------- | -------- | ----- | ----- |
| Decision Tree       | 72.0%    | 0.7205   | -0.12    | 0.28  | 0.28  |
| Random Forest       | 82.4%    | 0.8120   | 0.297    | 0.176 | 0.176 |
| Logistic Regression | 79.0%    | 0.7843   | 0.160    | 0.21  | 0.21  |
| XGBoost             | 82.4%    | 0.8126   | 0.295    | 0.176 | 0.176 |

## How It Works
- Data Preprocessing: Convert RNA sequences → numerical ASCII features.
- Train-Test Split: 50-50 split for evaluation.
- Model Training: Decision Tree, Random Forest, Logistic Regression, and XGBoost.
- Evaluation: Compare models using multiple metrics (accuracy, F1, R², ROC).

## Contributors
Ch. Thilak Adithya 
S. Shyam Prasad 
Ch. Anoop Rao 
S. Hari Prasad 

