# House Price Prediction Project

Group 40 — DA 591, Spring 2026

## Description

This project predicts house prices using the King County Housing dataset. We cleaned the data, did exploratory data analysis to understand the features, and then trained several machine learning models to find the best one for predicting house prices.

We tried Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost. After tuning the models and applying log transformation on the target variable, our best model was LightGBM with log transform which got a Test R² of 0.8958 with only 2.7% overfitting. We also validated the results using 5 fold cross validation and analyzed the prediction errors by price range using MAPE.

## Project Files

| File | Description |
|---|---|
| 01_Data_Cleaning.ipynb | Loading the raw data, handling missing values, removing duplicates, fixing data types, and saving the cleaned dataset |
| 02_EDA.ipynb | Exploratory data analysis, visualizations, correlation analysis, and feature selection (15 features selected) |
| 03_Model_Building.ipynb | Training and tuning all models, log transformation, MAPE analysis, and 5 fold cross validation |
| kc_house_data.csv | The original raw dataset from Kaggle |
| cleaned_house_data.csv | The cleaned dataset produced by notebook 01 |
| selected_features.json | The 15 features selected during EDA |
| requirements.txt | Python packages needed to run the code |

## How to Run

1. Clone this repository or download all the files into the same folder

2. Install the required packages by running:
```
pip install -r requirements.txt
```

3. Run the notebooks in order:
   - First run `01_Data_Cleaning.ipynb` (this produces cleaned_house_data.csv)
   - Then run `02_EDA.ipynb` (this does the analysis and saves selected features)
   - Then run `03_Model_Building.ipynb` (this trains the models and shows results)

You can also run the install cell at the top of notebook 03 which installs everything for you.

## Dependencies

- Python 3.9 or higher
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- catboost

All dependencies can be installed using `pip install -r requirements.txt`

## Dataset

The King County Housing dataset contains 21,613 house sale records from King County, Washington (which includes Seattle). It has 21 features including things like number of bedrooms, square footage, location, condition, and grade.

Source: Kaggle (https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
