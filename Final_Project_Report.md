# House Price Prediction and Analysis Using King County Housing Data

**Course:** DA 591 – Capstone Project

**Team:** Ashwin, Ashwath, Namrata Mane (Project #40)

---

## 1. Introduction

### 1.1 Problem Definition

Buying a home is one of the most significant financial decisions an individual can make. Sellers want to list their properties at the right price, buyers want to know if they are getting a fair deal, and real estate agents need reliable pricing guidance to serve their clients well. Despite the availability of housing data, accurately estimating the value of a property remains difficult because house prices are influenced by dozens of interrelated factors such as square footage, location, construction quality, and the age of the building (Kaggle, 2016).

This project set out to answer a straightforward question: given a set of property attributes, can we build a machine learning model that reliably predicts the sale price of a house in King County, Washington? The goal was to develop and compare several regression models, apply data transformation techniques, and identify the single best-performing model for this task.

### 1.2 Background Information

King County is the most populous county in the state of Washington and includes the city of Seattle. The local housing market is well documented, which makes it a practical case study for predictive analytics. The dataset used in this project was originally compiled from public records of home sales that took place between May 2014 and May 2015 and was published on Kaggle (Kaggle, 2016). Prior work on this dataset has typically relied on a single model, most commonly Linear Regression, and has not always explored the impact of target variable transformations or systematic hyperparameter tuning. Our project extends this prior work by comparing five different regression algorithms and by applying a log transformation to the target variable, which addressed the heavy right skew in the price distribution and led to measurable improvements in model accuracy.

---

## 2. Methodology

### 2.1 Dataset Description

The raw dataset contained 21,613 records and 21 columns. Each record represented a single home sale in King County. The target variable was the sale price, which ranged from $75,000 to $7,700,000 with a median of $450,000 and a mean of approximately $540,088. The standard deviation was $367,315, which is large relative to the mean and indicates substantial spread in the data. The 20 predictor columns included physical attributes of the property (bedrooms, bathrooms, square footage of the living space and lot, number of floors), quality indicators (grade on a scale of 1 to 13, condition on a scale of 1 to 5), temporal information (year built, year renovated), geographical coordinates (latitude, longitude, zipcode), and neighborhood reference values (average square footage of the nearest 15 neighbors). The dataset had no missing values in its original form.

### 2.2 Data Cleaning

Data cleaning was carried out in a dedicated Jupyter Notebook (01_Data_Cleaning.ipynb). The notebook was structured into 14 systematic steps, each documented with markdown cells explaining the rationale behind every decision. The following operations were performed:

1. **Library import.** We loaded the Pandas and NumPy libraries to handle all data manipulation tasks.

2. **Dataset loading.** The raw CSV file `kc_house_data.csv` was loaded into a Pandas DataFrame. The initial dimensions were confirmed as 21,613 rows and 21 columns.

3. **Initial inspection.** We examined the first and last five rows of the dataset using `.head()` and `.tail()`, and printed the column names to develop an understanding of the available features.

4. **Data type verification.** Using `.info()` and `.dtypes`, we confirmed that all numerical columns had the correct integer or float data types. The date column was stored as a string (object type), which would need conversion.

5. **Statistical summary.** We ran `.describe()` to generate summary statistics for all numerical columns, including the count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum for each feature. This step revealed key distributional characteristics, such as the minimum price of $75,000 and maximum of $7,700,000.

6. **Missing value check.** We computed `df.isnull().sum()` for each column and confirmed that the total number of missing values across the entire dataset was zero. Therefore, no imputation was required.

7. **Duplicate record detection.** We checked for exact duplicate rows using `df.duplicated().sum()` (result: 0 exact duplicates). We then checked for duplicate property IDs using `df['id'].duplicated().sum()` and discovered 177 houses that were sold more than once during the dataset's time period.

8. **Duplicate handling.** To resolve the 177 duplicate property IDs, we first sorted the dataset by date in descending order using `df.sort_values('date', ascending=False)`. We then kept only the first occurrence of each property ID (which corresponds to the most recent sale) using `df.drop_duplicates(subset='id', keep='first')` and reset the index. After this operation, the dataset contained 21,436 records.

9. **Date column conversion.** The date column, originally stored as a string in the format `20141013T000000`, was converted to a proper datetime type using `pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')`. From this, we extracted two new numerical columns: `sale_year` and `sale_month`. The data covered sales from May 2, 2014 to May 27, 2015.

10. **Unusual value detection.** We examined the bedroom distribution and found one record with 33 bedrooms but only 1,620 sqft of living space, which yielded approximately 49 sqft per bedroom — far below the reasonable minimum of 100 sqft per bedroom. This was clearly a data entry error. We also identified 13 properties with 0 bedrooms, which could represent studios, and 10 properties with 0 bathrooms. Additionally, we verified that all `sqft_living` values were positive (minimum 290, maximum 13,540).

11. **Data error correction.** The 33-bedroom entry was corrected to 3 bedrooms using a targeted mask: `(df['bedrooms'] == 33) & (df['sqft_living'] < 2000)`. The 0-bedroom and 0-bathroom entries were retained as they likely represent studios or unconventional properties. This specific correction was printed and logged in the notebook output for transparency.

12. **Price outlier analysis.** We computed the Interquartile Range (IQR) for the price column: Q1 = $324,866, Q3 = $645,000, IQR = $320,134. The resulting outlier boundaries were: lower bound = −$155,335 (no houses below this) and upper bound = $1,125,201 (1,143 houses above this, representing 5.33% of the data). We made the deliberate decision to retain all price data because the extreme values represent real market segments such as luxury homes and waterfront properties. The notebook documented this decision explicitly and noted that the choice could be revisited if model performance was poor.

13. **Feature engineering.** We created seven new columns from the existing data:
    - `house_age` — calculated as `sale_year - yr_built`. The resulting ages ranged from −1 (a house listed before its build year, likely pre-construction) to 115 years, with an average of 43.2 years.
    - `renovated` — a binary flag (1 or 0) derived from `yr_renovated > 0`. Only 4.25% of houses (910 out of 21,436) had been renovated.
    - `price_per_sqft` — calculated as `price / sqft_living`. Values ranged from $87.59/sqft to $810.14/sqft with an average of $264.72/sqft. This column was deliberately excluded from modeling because it is derived directly from the target variable and would cause data leakage.
    - `has_basement` — a binary flag derived from `sqft_basement > 0`. About 39.28% of houses (8,421) had a basement.
    - `total_rooms` — the sum of `bedrooms + bathrooms`. Values ranged from 0 to 16.5 with an average of 5.5.
    - `sale_year` and `sale_month` — extracted from the parsed datetime column to capture potential temporal trends.

14. **Final verification and export.** We performed a final check confirming the cleaned dataset had 21,436 rows and 28 columns, zero missing values, and zero duplicate IDs. The data types were verified one last time. The cleaned dataset was saved to `cleaned_house_data.csv` and then immediately reloaded to verify the export was successful.

### 2.3 Data Preparation (Exploratory Data Analysis)

Exploratory Data Analysis was conducted in a second Jupyter Notebook (02_EDA.ipynb). The cleaned dataset was loaded and its structure was confirmed as 21,436 entries with 28 columns. The EDA process consisted of the following steps:

1. **Price distribution analysis.** We plotted the house price distribution as a histogram and a bar chart of price ranges. The histogram showed a heavy right skew with the mean ($541,650) significantly exceeding the median ($450,000). We visualized both mean and median lines on the histogram. The bar chart categorized houses into price bands: 0–200K, 200–400K, 400–600K, 600–800K, 800K–1M, and 1M+. The 200–400K range contained the most houses.

2. **Categorical feature mapping.** Before visualization, we created a copy of the dataframe (`df_viz`) and mapped binary features (`waterfront`, `renovated`, `has_basement`) from 0/1 to "No"/"Yes" for more readable charts.

3. **Numerical feature distributions.** We plotted histograms for nine key numerical features: `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`, `bedrooms`, `bathrooms`, `floors`, `house_age`, and `price_per_sqft`. These plots revealed that most features were right-skewed, with a few outliers in the upper ranges.

4. **Categorical and binary feature analysis.** We created bar charts showing the distribution of binary features (waterfront, renovated, has_basement) and ordinal features (condition, view, grade). This revealed that the vast majority of houses are non-waterfront (99.2%), non-renovated (95.7%), and have an average grade of 7.

5. **Correlation analysis.** We computed the Pearson correlation coefficient between every numerical feature and the sale price. The results were visualized as a horizontal bar chart showing features sorted by their absolute correlation with price. We set a correlation threshold of 0.3 and used it as a guideline for feature selection. Features with strong positive correlations included `sqft_living` (0.70), `grade` (0.67), `sqft_above` (0.61), `bathrooms` (0.53), and `sqft_living15` (0.59). Features with moderate negative correlation included `long` (−0.32), meaning that more westward properties tended to be more expensive within King County.

6. **Scatter plot analysis.** We created scatter plots for key features against price to visually confirm the relationships detected by correlation analysis. These showed clear positive trends between price and variables such as `sqft_living` and `grade`.

7. **Feature selection.** Based on the correlation analysis, domain knowledge, and scatter plot examination, we selected 15 features for the modeling phase: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_basement`, `yr_built`, `yr_renovated`, `lat`, `long`, and `sqft_living15`. These features were saved to a JSON file (`selected_features.json`) for use in the modeling notebook. All 15 features come from the original dataset columns; the engineered features (such as `house_age`, `renovated`, `has_basement`, `total_rooms`) did not improve model performance when tested alongside the originals. This is likely because tree-based models can learn equivalent relationships from the raw inputs on their own (Chen and Guestrin, 2016).

### 2.4 Model Creation and Iteration

All modeling work was carried out in a third Jupyter Notebook (03_Model_Building.ipynb). The notebook followed a structured progression from a simple baseline model to advanced gradient boosting algorithms, with systematic hyperparameter tuning and target variable transformation at each stage.

**Data splitting and scaling.** The cleaned data was loaded and the 15 selected features were extracted. The data was split into 80% training (17,148 samples) and 20% testing (4,288 samples) using `train_test_split` with a fixed `random_state=42` to ensure reproducibility. Features were standardized using `StandardScaler`, which was fitted on the training set and applied to both training and testing sets to prevent data leakage.

**Model 1: Linear Regression.** We began with a standard Linear Regression from scikit-learn as a simple, interpretable baseline. This model has no hyperparameters to tune. Training R² was 0.6927 and Test R² was 0.7044, indicating an unusual negative overfitting gap of −0.0118 (test slightly outperformed training). The Test MAE was $128,393 and the Test RMSE was $198,011. We also examined the feature coefficients: `sqft_living` had the largest positive impact (+$172,034 per standard deviation increase), followed by `grade` (+$110,746), while `yr_built` had the largest negative effect (−$75,115, meaning older homes tend to be cheaper). The model was saved as `linear_regression_model.pkl` using joblib.

**Model 2: Random Forest.** Since the 70% R² from Linear Regression was insufficient, we moved to a Random Forest Regressor with 100 estimators (trees) and the default hyperparameters. Training R² jumped to 0.9834 while Test R² was 0.8590, yielding a large overfitting gap of 0.1244. The Test MAE improved significantly to $70,436.

**Tuning Random Forest with GridSearchCV.** To reduce overfitting, we performed hyperparameter tuning using GridSearchCV with 3-fold cross-validation. The parameter grid searched was:
- `max_depth`: [5, 8, 10, 12]
- `min_samples_split`: [10, 20, 50]
- `min_samples_leaf`: [5, 10, 20]

The best parameters found were `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=5`. After tuning, Training R² decreased to 0.9280 and Test R² to 0.8565, reducing the overfitting gap from 0.1244 to 0.0715 — a reduction of 0.0529. However, the test score dropped slightly.

**Model 3: XGBoost.** We then trained an XGBRegressor with `n_estimators=100`, `learning_rate=0.1`, and `max_depth=6`. The base model achieved Training R² of 0.9544 and Test R² of 0.8871, with a MAE of $67,873 and an overfitting gap of 0.0673.

**Tuning XGBoost.** Grid search was performed with the following parameter grid:
- `max_depth`: [4, 5, 6]
- `min_child_weight`: [1, 3, 5]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]

The best parameters found were `max_depth=6`, `min_child_weight=1`, `subsample=0.8`, `colsample_bytree=0.8`. The tuned model achieved Test R² of 0.8922, a slight improvement over the base XGBoost.

**Model 4: LightGBM.** A LGBMRegressor was trained with `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`, and `verbose=-1` to suppress iteration output. Training R² was 0.9328 and Test R² was 0.8857, with a MAE of $68,037 and an overfitting gap of 0.0471.

**Tuning LightGBM.** The parameter grid for GridSearchCV included:
- `max_depth`: [4, 5, 6]
- `num_leaves`: [20, 31, 50]
- `min_child_samples`: [10, 20, 30]
- `subsample`: [0.8, 0.9, 1.0]

The best parameters found were `max_depth=6`, `num_leaves=31`, `min_child_samples=10`, `subsample=0.8`. The tuned model yielded Test R² of 0.8841, marginally lower than the base LightGBM, suggesting the defaults were already well suited for this data.

**Model 5: CatBoost.** A CatBoostRegressor was trained with `iterations=100`, `learning_rate=0.1`, `depth=6`, and `verbose=0`. Training R² was 0.9116 and Test R² was 0.8862, with a MAE of $67,649 and an overfitting gap of 0.0254. CatBoost tuning was also performed but resulted in a slightly lower Test R² of 0.8839.

**Log Transformation.** The price distribution was heavily right-skewed, with most homes priced below one million dollars and a long tail extending to $7.7 million. To address this, we applied a natural log transformation to the target variable using the `log1p` function and trained all five models again with this transformed target. Predictions were converted back to dollar values using the `expm1` function before evaluation.

The log transformation improved performance for all tree-based models. The most notable improvement was in LightGBM, which went from an R² of 0.8857 on the base model to 0.8958 with the log-transformed target — our highest single-split score. Random Forest improved from 0.8590 to 0.8778. XGBoost improved from 0.8871 to 0.8933. CatBoost saw a marginal change from 0.8862 to 0.8854. Linear Regression, however, dropped dramatically from 0.7044 to 0.4773. This happened because when log-transformed predictions are converted back to dollar values through exponentiation, the errors from a straight-line model are amplified significantly for high-value properties.

**Final Validation with 5-Fold Cross-Validation.** To confirm that the best model was not simply benefiting from a favorable train-test split, we ran 5-fold cross-validation on all models. In 5-fold cross-validation, the entire dataset is divided into five equal parts. The model is trained and tested five separate times, each time using a different part as the test set and the remaining four parts for training. The average score across all five runs provides a more robust estimate of real-world performance.

We ran 5-fold CV for all model variants: base models, log-transformed models, and both base and log versions of each algorithm. The cross-validation summary showed:

| Model | Avg Train R² | Avg Test R² | Avg Gap |
|---|---|---|---|
| LightGBM (Log) | 0.9224 | 0.9004 | 0.0220 |
| XGBoost (Log) | 0.9328 | 0.8999 | 0.0330 |
| LightGBM | 0.9328 | 0.8816 | 0.0512 |
| XGBoost | 0.9538 | 0.8805 | 0.0732 |
| CatBoost | 0.9116 | 0.8751 | 0.0365 |
| Random Forest | 0.9828 | 0.8727 | 0.1101 |
| Linear Regression | 0.6954 | 0.6934 | 0.0020 |

LightGBM with log transformation achieved the best average Test R² of 0.9004 with the smallest gap of 2.2%, confirming it generalizes well and is not overfitting. By contrast, Random Forest had the largest gap at 11%, indicating it memorized training data more heavily than the boosting methods.

---

## 3. Results

### 3.1 Numeric Results

The table below summarizes the test R-squared scores at each stage of the project for all five models.

| Model | Base R² | Tuned R² | Log Transform R² | 5-Fold CV Avg R² | Train-Test Gap |
|---|---|---|---|---|---|
| Linear Regression | 0.7044 | — | 0.4773 | 0.6934 | 0.20% |
| Random Forest | 0.8590 | 0.8565 | 0.8778 | 0.8727 | 11.01% |
| XGBoost | 0.8871 | 0.8922 | 0.8933 | 0.8999 | 3.30% |
| LightGBM | 0.8857 | 0.8841 | **0.8958** | **0.9004** | **2.20%** |
| CatBoost | 0.8862 | 0.8839 | 0.8854 | 0.8751 | 3.65% |

The champion model is LightGBM with log transformation, which explains approximately 90% of the variance in house prices. Its Mean Absolute Error on the test set is approximately $66,600, meaning that on average the model's prediction deviates by about $66,600 from the true sale price. Because the log transformation was applied, this error scales proportionally with the price: cheaper homes in the $200K range have dollar errors closer to $20K–$25K, while luxury homes worth several million may have errors of $200K or more. The MAE of $66,600 is reasonable in the context of an average sale price exceeding $540,000, placing the typical prediction error at roughly 12% of the average price.

### 3.2 Data Visualization

Correlation analysis revealed that the features most strongly correlated with price were `sqft_living` (0.70), `grade` (0.67), `sqft_above` (0.61), `bathrooms` (0.53), and `sqft_living15` (0.59). Feature importance analysis from the LightGBM champion model showed that geographic location (latitude and longitude combined) was by far the most influential predictor, accounting for approximately 100% of the relative importance (when normalized). This was followed by lot size (55%), living area square footage (45%), year built (42%), and grade (22%). This finding aligns with the well-known real estate principle that location is the primary driver of property value — two identical homes can differ dramatically in price depending on whether they are located in downtown Seattle or in a rural area of King County.

The price distribution was visualized before and after the log transformation. The original distribution was heavily right-skewed, with the majority of sales clustering below $1 million and a long tail of high-value properties. After applying the log transformation, the distribution became approximately normal, which improved the performance of the tree-based models by allowing them to learn more effectively across the full price spectrum.

---

## 4. Discussion and Contribution

### 4.1 Interpretation of Results

The results demonstrate that tree-based ensemble methods significantly outperform traditional Linear Regression for house price prediction in this dataset. Linear Regression achieved an R-squared of 0.70, which means it could explain only 70% of the price variation. All four tree-based models exceeded 0.88 on their base configurations, and the best model reached 0.90 after log transformation and cross-validation.

The log transformation proved to be the single most impactful technique in the project. It addressed the skewed price distribution and allowed the models to learn more effectively across the full price spectrum. Interestingly, hyperparameter tuning had a relatively small effect — only XGBoost showed a measurable improvement (from 0.8871 to 0.8922), while the other tree-based models performed slightly worse after tuning. This suggests that modern gradient boosting libraries ship with default parameters that are already well optimized for typical tabular datasets.

The 5-fold cross-validation results confirmed the reliability of the champion model. LightGBM with log transformation maintained the highest average test score (0.9004) and the lowest overfitting gap (2.2 percent). By contrast, Random Forest had the largest gap at 11 percent, indicating that it memorized training data more than the boosting-based models. Linear Regression had the smallest gap (0.2%) but the lowest accuracy, illustrating the classic bias-variance tradeoff: a very simple model underfits the data but does not overfit.

### 4.2 Success Evaluation

The project successfully built a model that explains 90% of the variance in King County house prices using only 15 features. The Mean Absolute Error of approximately $66,600 is reasonable given that the average home in the dataset sold for over $540,000, placing the error at roughly 12% of the average sale price. For a practical application such as generating initial price estimates for a real estate listing platform, this level of accuracy would provide useful guidance to both sellers and buyers.

### 4.3 Contributions

This project makes the following contributions to the study of house price prediction:

1. **Systematic multi-model comparison.** Rather than relying on a single algorithm, we trained, tuned, and compared five different regression models under identical conditions, providing a clear picture of which approach works best for this type of data.

2. **Demonstration of log transformation impact.** We showed that applying a log transformation to the skewed target variable improved performance for tree-based models while degrading Linear Regression, and we provided a clear explanation of why this occurs. The transformation effectively normalizes the price distribution and causes the model's error to scale proportionally with the price, which is a more desirable property for real estate applications.

3. **Rigorous validation methodology.** We used both single train-test splits and 5-fold cross-validation to ensure that our reported results are robust and not artifacts of a lucky data split. The two-stage validation approach — 3-fold CV for hyperparameter tuning and 5-fold CV for final model evaluation — provides a thorough assessment of model reliability.

4. **Practical feature importance analysis.** Our analysis confirmed that location, as captured by latitude and longitude, is the dominant predictor of house prices, followed by physical size and construction quality. This aligns with domain knowledge but is now quantified through data-driven analysis.

5. **Thorough data cleaning pipeline.** The 14-step data cleaning process included duplicate detection based on property IDs, correction of data entry errors (e.g., 33-bedroom entry), IQR-based outlier analysis, and the creation of seven engineered features, all documented transparently in the notebook.

### 4.4 Future Work

Several directions could extend this project in meaningful ways:

- **Incorporating external data.** Adding neighborhood-level data such as school district ratings, crime statistics, and proximity to public transit could improve predictions for location-sensitive markets.
- **Temporal modeling.** The current model does not account for housing market trends over time. Incorporating macroeconomic indicators or training on a longer time series could help the model adapt to changing market conditions.
- **Advanced hyperparameter tuning.** Using Bayesian optimization or random search with a larger parameter space might yield further improvements beyond what GridSearchCV with a limited grid could achieve.
- **Deployment.** Packaging the champion model into a simple web application where users can input property attributes and receive an estimated price would demonstrate the practical value of the work.

---

## 5. Conclusion

This project developed and compared five regression models for predicting house sale prices in King County, Washington. Starting from a raw dataset of 21,613 records with 21 columns, we cleaned the data by removing 177 duplicate property listings, converting the date column, correcting a data entry error in the bedroom column, and engineering seven additional features. Through Exploratory Data Analysis, we computed Pearson correlations, visualized price distributions and feature relationships, and selected 15 meaningful predictors using a correlation threshold of 0.3 and domain knowledge.

We systematically evaluated Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost. Features were standardized using StandardScaler, and the data was split 80/20 with a fixed random seed. Through hyperparameter tuning with GridSearchCV (3-fold cross-validation) and application of a log transformation to the target variable using log1p/expm1, we identified LightGBM with log-transformed prices as the champion model. This model achieved an R-squared of 0.8958 on a single test split and an average R-squared of 0.9004 across 5-fold cross-validation, with a train-test gap of only 2.2%. The Mean Absolute Error was approximately $66,600. The project demonstrates that careful data preparation, target variable transformation, and rigorous cross-validation can yield highly accurate and generalizable house price predictions using readily available property data.

---

## Reference List

Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.

Kaggle (2016) *House Sales in King County, USA*. Available at: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction (Accessed: January 2026).

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.Y. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', *Advances in Neural Information Processing Systems*, 30, pp. 3146–3154.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V. and Gulin, A. (2018) 'CatBoost: Unbiased Boosting with Categorical Features', *Advances in Neural Information Processing Systems*, 31, pp. 6638–6648.

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J. and Kern, R. (2020) 'Array programming with NumPy', *Nature*, 585, pp. 357–362.

McKinney, W. (2010) 'Data Structures for Statistical Computing in Python', *Proceedings of the 9th Python in Science Conference*, pp. 56–61.

Hunter, J.D. (2007) 'Matplotlib: A 2D Graphics Environment', *Computing in Science and Engineering*, 9(3), pp. 90–95.
