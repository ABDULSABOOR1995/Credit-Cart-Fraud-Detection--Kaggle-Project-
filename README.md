# Credit Cart Fraud Detection (Kaggle Project)

## Context
Credit Card Fraud Detection is a Binary Classification ML Project announced on Kaggle competition. The contest behind the project is that credit card companies must be able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase

## Content
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.<br>

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, Kaggle didn't provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Inspiration
Identify fraudulent credit card transactions.

## Table of Content:
[1- Data Reading & Understanding](#h1)<br>
[2- Data Preparation](#h2)<br>
[3- Data Visualization](#h3)<br>
[4- Feature Normalization](#h4)<br>
[5- Data Selection](#h5)<br>
[6- Model Selection](#h6)<br>
>[6.1. Random Forest Model](#h6.1)<br>
[6.2. XGBoost](#h6.2)<br>

[7- Summerize Models with their results](#h7)<br>

[Conclusion](#conclusion)

### 1- Data Reading & Understanding:
First we read <b>creditcard.csv</b> file in pandas and create its data frame and then we start understanding data by applying basic pandas statistical methods on the data frame.


