# Sberbank Russian Housing Market Prediction

This project explores the Sberbank Russian Housing Market dataset to predict realty prices based on various housing and macroeconomic features. It was developed as part of the CISC/CMPE 251 Data Analytics course.

## 👥 Team Members
* Drew Tessmer
* Ethan Doma
* Alex Lester
* Zuhair Abbas

## 🚀 Project Overview
The objective of this project is to develop a robust machine learning pipeline to predict residential property prices in Moscow. The dataset includes a wide array of features, from property-specific details (e.g., square footage, floor, building year) to neighbourhood characteristics and broader macroeconomic indicators.

## Files
* CISC251_Deliverable3_Complete.ipynb - Jupyter notebook of complete ML pipeline
* CISC251_Deliverable3_Complete.pdf - PDF report of Jupyter notebook
* data.zip - Zip folder of all kaggle csv data 
* requirements.txt - List of used modules and libraries for entire project 

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
* **Models:** K-Nearest Neighbors (KNN), Ridge Regression, Decision Tree Regressor

## 📊 Methodology

### 1. Data Preprocessing
* **Feature Selection:** Dropped columns with >50% missing values and removed identifiers like `id` and `timestamp`.
* **Numerical Features:** Imputed missing values using the median and applied `StandardScaler` for normalization.
* **Categorical Features:** Imputed missing values with a constant "MISSING" label and used `OneHotEncoder` to transform categories into binary features.

### 2. Model Development & Comparison
We evaluated three baseline models using 5-fold cross-validation:
* **Ridge Regression:** Achieved the highest accuracy and stability (RMSE: ~3.43M, R²: 0.45).
* **KNN Regressor:** Captured local patterns but was sensitive to outliers.
* **Decision Tree Regressor:** Showed high variance with significant errors on specific predictions.

### 3. Hyperparameter Tuning
Focused systematic tuning was performed on the KNN model using `GridSearchCV`:
* **Optimal Hyperparameters:** `n_neighbors: 10`, `weights: 'distance'`, `metric: 'euclidean'`.
* **Analysis:** Tuning revealed that while `k=10` was localy optimal, the default parameters were already quite effective for this high-dimensional dataset.

## 📈 Results
The final model performance was validated against Kaggle leaderboards:
* **Kaggle Public Score:** 0.40813 (RMSLE)
* **Kaggle Full Score:** 0.42293 (RMSLE)

## 📁 Repository Structure
* `CISC251_Deliverable3_Complete.ipynb`: Full analytical workflow and model training.
* `data/raw/`: Contains `train.csv`, `test.csv`, and `macro.csv`
* `data_dictionary.txt`: Descriptions of the features used in the dataset.

## 📝 Conclusion
Ridge Regression emerged as the most reliable model for this dataset due to its ability to handle feature correlation through L2 regularization. Future work could involve more complex ensemble methods like XGBoost or Random Forests to capture non-linear relationships more effectively.
