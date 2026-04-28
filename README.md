# House Price Prediction and Analysis Using King County Housing Data

**Course:** DA 591 – Data Science & Business Analytics Practicum

**Team:** Ashwin Santhanakrishnan, Aswath Santhanakrishnan, Namrata Mane (Project #40)

---

## Description

This project predicts house sale prices in King County, Washington using machine learning. The code is organized into three Jupyter Notebooks that follow the standard data science pipeline:

1. **01_Data_Cleaning.ipynb** — Loads the raw King County housing dataset (21,613 records, 21 columns), removes 177 duplicate property listings, converts the date column from string to datetime, corrects a data entry error (33 bedrooms changed to 3), performs IQR-based outlier analysis on the price column, and engineers seven new features (house_age, renovated, price_per_sqft, has_basement, total_rooms, sale_year, sale_month). Outputs a cleaned CSV file (`cleaned_house_data.csv`) with 21,436 records and 28 columns.

2. **02_EDA.ipynb** — Performs exploratory data analysis on the cleaned dataset. Generates price distribution histograms, numerical feature distributions, categorical feature bar charts, Pearson correlation analysis, and scatter plots. Selects 15 features for modeling based on correlation with price (threshold of 0.3) and domain knowledge. Saves the selected feature list to `selected_features.json`.

3. **03_Model_Building.ipynb** — Trains and compares five regression models: Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost. Performs hyperparameter tuning using GridSearchCV with 3-fold cross-validation for each tree-based model. Applies a log transformation to the target variable to handle price skewness. Runs 5-fold cross-validation on all model variants. The champion model is LightGBM with log transformation, achieving an average R² of 0.9004 across 5-fold CV with a train-test gap of 2.2%.

---

## How to Run

### Step 1: Clone the repository
```
git clone https://github.com/AswathSanthanakrishnan/House-Price-Prediction-Project---Final-sem-project-.git
cd House-Price-Prediction-Project---Final-sem-project-
```

### Step 2: Install dependencies
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost joblib
```

### Step 3: Run the notebooks in order
Open Jupyter Notebook or JupyterLab and run each notebook sequentially:

1. Run **01_Data_Cleaning.ipynb** first — this produces `cleaned_house_data.csv`
2. Run **02_EDA.ipynb** second — this reads the cleaned data and produces `selected_features.json`
3. Run **03_Model_Building.ipynb** last — this reads the cleaned data, trains all models, and outputs the results

Each notebook must be run in order because later notebooks depend on files produced by earlier ones.

---

## Dependencies

| Package | Version Used | Purpose |
|---|---|---|
| Python | 3.9+ | Programming language |
| pandas | 2.0+ | Data loading, cleaning, and manipulation |
| numpy | 1.24+ | Numerical operations |
| matplotlib | 3.7+ | Data visualization (histograms, scatter plots) |
| seaborn | 0.12+ | Statistical data visualization (correlation heatmaps) |
| scikit-learn | 1.3+ | Linear Regression, Random Forest, StandardScaler, train_test_split, GridSearchCV, cross_val_score, evaluation metrics |
| xgboost | 2.0+ | XGBRegressor for gradient boosting |
| lightgbm | 4.0+ | LGBMRegressor for gradient boosting |
| catboost | 1.2+ | CatBoostRegressor for gradient boosting |
| joblib | 1.3+ | Saving trained model files (.pkl) |

---

## Project Structure

```
├── kc_house_data.csv                 # Raw dataset (21,613 records)
├── cleaned_house_data.csv            # Cleaned dataset (21,436 records)
├── selected_features.json            # 15 selected features from EDA
├── 01_Data_Cleaning.ipynb            # Notebook 1: Data cleaning and feature engineering
├── 02_EDA.ipynb                      # Notebook 2: Exploratory data analysis
├── 03_Model_Building.ipynb           # Notebook 3: Model training, tuning, and evaluation
├── linear_regression_model.pkl       # Saved Linear Regression model
├── Final_Project_Report.md           # Final project report
├── Group_Contribution_Report_Ashwath.md  # Individual contribution report
└── README.md                         # This file
```
