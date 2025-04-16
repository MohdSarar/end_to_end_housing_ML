#  California Housing Price Prediction – My End-to-End ML Pipeline

Hello!  I'm Mohammed, and this project is part of my personal portfolio where I built a complete end-to-end machine learning pipeline to **predict median house prices in California** using the famous California Housing dataset.

I used this project to reinforce my skills in **data cleaning, EDA, feature engineering, model training and evaluation**. It includes a fully documented pipeline coded in Python with Scikit-learn, Matplotlib, and Seaborn.

---

##  Project Overview

This repo includes the following steps:

-  Loading and inspecting the California Housing dataset
-  Data cleaning and initial analysis
-  Exploratory Data Analysis (EDA): histograms, boxplots, scatterplots, heatmaps, and pairplots
-  Feature scaling using `StandardScaler`
-  Training two models:
  - Linear Regression
  - Random Forest Regressor
-  Evaluation using RMSE and R²
-  Overfitting check (train vs test performance)
-  Feature importance analysis (for Random Forest)
-  Saving trained models and scaler using `joblib`

---

##  Why I Did This

I wanted to go beyond theory and actually apply the entire data science process from A to Z. This project allowed me to:

- Practice **exploratory data analysis** using visualizations.
- Understand **how feature scaling** affects Linear Regression vs. Random Forest.
- Compare **model performance** and detect overfitting.
- Learn how to **persist models** for future use.

---
## 📸 Visual Explorations (Shown in Notebook)

During the exploratory phase, I generated several visualizations including:

-  A correlation heatmap between all features and the target
-  Scatterplots to explore linear/non-linear trends
-  Feature importance ranking using Random Forest
-  RMSE comparison on train vs test set to detect overfitting

These plots are generated dynamically and shown during script execution.


##  How to Use This Project

### 1. Clone the repository

```bash
git clone https://github.com/MohdSarar/end_to_end_housing_ML.git
cd end_to_end_housing_ML
