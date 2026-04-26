# Group Contribution Report

**Name:** Ashwath

**Course:** DA 591 – Capstone Project

**Group:** Project #40

---

## Summary of My Contributions

### Research

I did the initial research on what machine learning algorithms would work well for a regression problem like house price prediction. I looked into Linear Regression as a baseline and then researched tree-based models including Random Forest, XGBoost, LightGBM, and CatBoost. I also researched the effect of log transformations on skewed target variables, which ended up being one of the key techniques that improved our model's accuracy.

### Code

I was responsible for writing the majority of the Python code across all three Jupyter Notebooks:

- **01_Data_Cleaning.ipynb** — I wrote the full data cleaning pipeline, which included loading the raw dataset, checking for missing values and duplicates, handling the 177 duplicate property IDs by keeping the most recent sale, converting the date column from string to datetime, identifying and fixing the data entry error (33-bedroom house), running the IQR outlier analysis on prices, and engineering seven new features (house_age, renovated, price_per_sqft, has_basement, total_rooms, sale_year, sale_month). I also wrote the code to export the cleaned data to a CSV file.

- **02_EDA.ipynb** — I wrote the code for the exploratory data analysis notebook. This included the price distribution histograms, bar charts for categorical features, the Pearson correlation analysis, scatter plots for key features vs price, and the feature selection logic where we chose 15 features based on a correlation threshold of 0.3 and domain knowledge.

- **03_Model_Building.ipynb** — I wrote all the modeling code. This covered the train-test split (80/20), feature scaling with StandardScaler, training all five models (Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost), hyperparameter tuning with GridSearchCV for each tree-based model, implementing the log transformation using log1p and expm1, and running 5-fold cross-validation on all model variants to identify the champion model.

### Data Analysis

I performed the main data analysis work, including interpreting the correlation results, analyzing the feature importance outputs from the tree-based models, comparing overfitting gaps across models, and determining that LightGBM with log transformation was the best model based on both single-split and cross-validation results.

### Software and Tests

I set up the Python environment and made sure all the necessary libraries were installed (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, catboost). I ran all the model training and evaluation code, and I also ran the final 5-fold cross-validation tests that confirmed LightGBM as the champion model with an average R-squared of 0.9004.

### Reports and Presentations

I helped write sections of the final project report, specifically the Methodology section (data cleaning, EDA, and model creation details) and the Results section with the comparison tables. For the final presentation, I created slides covering the model iteration process, log transformation results, cross-validation methodology, numeric results, and feature importance analysis. I also presented those slides during the final presentation.

### GitHub

I managed the GitHub repository for the project. I pushed the cleaned data files, all three notebooks, and the presentation materials to the repo. I also made sure the README was updated so anyone looking at the repository could understand the project structure.

---

## Group Member Discussion and Ratings

### Ashwin

Ashwin contributed to the project by helping with the initial dataset research and the introduction and background sections of the final report. He also worked on the problem definition slide for the presentation and helped review the EDA notebook outputs. During the presentation, he covered the introduction and problem definition slides. We worked well together and communicated regularly through group meetings. He was responsive when tasks were assigned and completed his portions on time.

**Rating: 4/5**

### Namrata Mane

Namrata contributed by working on the discussion and conclusion sections of the final report. She also helped with the data visualization slides for the presentation and presented those slides during the final presentation. She participated in group discussions and gave useful feedback on the model comparison results. We did not have any major issues working together, and she was a reliable team member throughout the project.

**Rating: 4/5**
