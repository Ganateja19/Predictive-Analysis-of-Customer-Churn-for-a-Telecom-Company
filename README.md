# AI-Powered Churn Prediction System for Telecom Customer Retention

Project Overview:
The objective of this project was to develop a machine learning model to predict customer churn using the Telco Customer Churn dataset. The dataset contains 7,043 customer records and 21 features, including demographics, service details, and billing information.
Churn prediction is critical in the telecom industry, as retaining customers is much more cost-effective than acquiring new ones. This project aimed to build a predictive model and analyze the key factors that influence churn.


Data Preprocessing
1.	Loaded the dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv).
2.	Checked for duplicate records and verified that none existed.
3.	Converted TotalCharges from object to numeric; handled 11 missing values by dropping those rows.
4.	Created a tenure_group column by binning tenure into 12-month intervals to simplify tenure interpretation.
5.	Dropped non-predictive columns like customerID as it holds no useful information for modeling.
6.	Analyzed categorical and numerical columns separately to understand value distributions.
7.	One-hot encoded all categorical features to prepare the data for model input.
8.	Standardized numerical columns (MonthlyCharges, TotalCharges) for future PCA-based modeling.
9.	Split the dataset into features (X) and target (y = Churn).
10.	Verified there were no remaining missing values post-transformation using isnull().sum().
11.	Loaded the dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv).
12.	Converted TotalCharges to numeric and handled 11 missing values using row removal.
13.	Created a tenure_group column by binning tenure into 12-month intervals.
14.	Dropped non-predictive columns like customerID.
15.	One-hot encoded categorical features.
16.	Split the data into features (X) and target (y = Churn).



Handling Class Imbalance
To address class imbalance, we used SMOTEENN, which combines Synthetic Minority Oversampling Technique (SMOTE) with Edited Nearest Neighbors (ENN). This helps both oversample the minority class and clean noisy examples.
Before applying SMOTEENN, the target variable distribution was significantly imbalanced:
Churn = 0 (No Churn): 5174
Churn = 1 (Churn): 1869
This imbalance can bias the model to favor the majority class (no churn), reducing the model’s ability to detect churners.
We chose SMOTEENN over simpler resampling strategies like random oversampling or SMOTE alone because:
•	SMOTE creates synthetic minority class examples that help balance the dataset.
•	ENN removes ambiguous or overlapping samples from both classes, improving class separation.
