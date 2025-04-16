# california_housing_ml_pipeline_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

sns.set(style="whitegrid")

# 1. Load dataset
print("Loading California housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')
df = pd.concat([X, y], axis=1)

# 2. Initial Inspection
print("\nInitial Data Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# 3. EDA - Histograms for all features
print("Plotting histograms...")
df.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# 4. EDA - Boxplots to detect outliers
print("Plotting boxplots for outlier detection...")
plt.figure(figsize=(15, 10))
for i, column in enumerate(X.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df[column], color='skyblue')
plt.tight_layout()
plt.show()

# 5. EDA - Correlation Heatmap
print("Plotting correlation matrix...")
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.show()

# 6. EDA - Target vs Feature scatterplots
print("Plotting scatterplots: Target vs Features...")
plt.figure(figsize=(16, 12))
for i, column in enumerate(X.columns):
    plt.subplot(3, 3, i + 1)
    sns.scatterplot(x=df[column], y=df['MedHouseVal'], alpha=0.3)
    plt.title(f"{column} vs MedHouseVal")
plt.tight_layout()
plt.show()

# 7. EDA - Pairplot (only top correlated features to target)
top_corr_features = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False)[1:5].index.tolist()
sns.pairplot(df[top_corr_features + ['MedHouseVal']], diag_kind='kde')
plt.suptitle("Pairplot of Top Features vs MedHouseVal", y=1.02)
plt.show()

# 8. Train-Test Split
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']
print("Splitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Train Linear Regression
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# 11. Train Random Forest
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 12. Evaluation Function
def evaluate_model(model, X, y, name="Model"):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\n{name} Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return mse, r2


# 13. Feature Importance Plot
print("Plotting feature importances from Random Forest...")
importances = rf_model.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp, y=feat_imp.index, palette="viridis")
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# 14. Overfitting Test
print("Checking for overfitting...")

def plot_overfitting_check(model, X_train, X_test, y_train, y_test, model_name):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=[f'{model_name} Train', f'{model_name} Test'], 
                y=[train_rmse, test_rmse], palette='coolwarm')
    plt.ylabel('RMSE')
    plt.title(f'Overfitting Check: {model_name}')
    plt.ylim(0, max(train_rmse, test_rmse) * 1.2)
    plt.tight_layout()
    plt.show()

# Apply to both models
plot_overfitting_check(lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")
plot_overfitting_check(rf_model, X_train, X_test, y_train, y_test, "Random Forest")


# 15. Evaluate Models
evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
evaluate_model(rf_model, X_test, y_test, "Random Forest")

# 16. Save Models
print("Saving models and scaler...")
joblib.dump(lr_model, "linear_regression_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Full pipeline completed with EDA and model training.")
