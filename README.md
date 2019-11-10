# Credit Cart Fraud Detection (Kaggle Project)

## Context
Credit Card Fraud Detection is a Binary Classification ML Project announced on Kaggle competition. The contest behind the project is that credit card companies must be able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase

## Content
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.<br>

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, Kaggle didn't provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Inspiration
Identify fraudulent credit card transactions.

## Table of Content:
[1. Data Reading & Understanding](#h1)<br>
[2. Data Preparation](#h2)<br>
[3. Data Visualization](#h3)<br>
[4. Feature Normalization](#h4)<br>
[5. Data Selection](#h5)<br>
[6. Model Selection](#h6)<br>
>[6.1. Random Forest Model](#h6.1)<br>
[6.2. XGBoost](#h6.2)<br>

[7. Summerize Models with their results](#h7)<br>

[Conclusion](#conclusion)

### 1. Data Reading & Understanding:
First we read <b>"creditcard.csv"</b> file in pandas and create its data frame and then we start understanding data by applying basic pandas statistical methods on the data frame.

### 2. Data Preparation: 
The creditcard data was highly imbalanced. 99.83% of the transactions in this data set were not fraudulent while only 0.17% were fraudulent.Using the original data set would not prove to be a good idea for a very simple reason: Since over 99% of our transactions are non-fraudulent, an algorithm that always predicts that the transaction is non-fraudulent would achieve an accuracy higher than 99%. Nevertheless, that is the opposite of what we want. We do not want a 99% accuracy that is achieved by never labeling a transaction as fraudulent, we want to detect fraudulent transactions and label them as such.
To create our balanced training data set, I took all of the fraudulent transactions in our data set and counted them. Then, I randomly selected the same number of non-fraudulent transactions and concatenated the two. After shuffling this newly created data set, I decided to output the class distributions once more to visualize the difference.
<b> Note: </b> The sata set we created isn't completely balanced, The dataset contained 62.5% non-fraud transactions while 37.5% fraud transactions but it is good for making classification model. 

And the we make <b>train.csv</b> and <b>test.csv</b>.

### 3. Data Visualization:
After that we visualize data distribution between both classes with the help of different charts. We'll also make charts to visualize Correlation of all features with target variable(Class).

### 4. Feature Normalization:
Feature normalization makes the values of each feature in the data have zero-mean (when subtracting the mean in the numerator) and unit-variance. In this dataset we have only one feature named <b>"normalizedAmount"</b> having values greater than 1, except this feature, all other have values in range 0. So we apply feature normalization on <b>"normalizedAmount"</b>.

### 5. Data Selection:
Now we make subset of features that have high impact on target variable(class).

### 6. Model Selection:
From the past experience, we become to know that in Classification Algorithms, the best algorithms in terms of efficiency and accuracy are <b>Random Forest </b>and <b> XG Boost</b>, so we use both of them to make a model and then select the better one from both of them.

#### 6.1. Random Forest Model:
Now we are able to make ML model for classification purpose. First of all, we have choose <b> Random Forest</b> model. It is best supervised learning algorithm. It is considered as a highly accurate and robust method because of the number of decision trees participating in the process. So the process of creating model and training and testing model is given below:

First we'll apply <b> Grid Search</b> on Random Forest to find best hyperparametrs.

After finding best hyperparametrs, we'll apply cross validation on Random forsest to find <b> average accuray score, f1-score, roc-auc score and log-loss.


Then we extract features and make subset of best features by finding feature importance from random forest algo.


But features that I have extracted from random Forest algo(X11-X15) weren't giving result as good as  the features we extracted before by correlation(X1-X10). So I selected those features for training and testing purpose.

Now I trained and tested the model on training and testing data. It has given the following result.

After that I finalized the best <b> Random Forest Model</b> that is given below:

#### Confusion Matrix of Best RF Model:



