# Credit Cart Fraud Detection (Kaggle Project)

## Context
Credit Card Fraud Detection is a Binary Classification ML Project announced on Kaggle competition. The contest behind the project is that credit card companies must be able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase

## Content
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.<br>

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, Kaggle didn't provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Inspiration
Identify fraudulent credit card transactions.

<a id='back'></a>
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

<a id='h1'></a>
### 1. Data Reading & Understanding:
First we read <b>"creditcard.csv"</b> file in pandas and create its data frame and then we start understanding data by applying basic pandas statistical methods on the data frame.

<br><br><br>
![Screenshot_63](https://user-images.githubusercontent.com/46135898/68549133-b7f06c80-0416-11ea-9da3-94304628f9c2.png)
<br><br><br>

[Go Back](#back)

<a id='h2'></a>
### 2. Data Preparation: 
The creditcard data was highly imbalanced. 99.83% of the transactions in this data set were not fraudulent while only 0.17% were fraudulent.Using the original data set would not prove to be a good idea for a very simple reason: Since over 99% of our transactions are non-fraudulent, an algorithm that always predicts that the transaction is non-fraudulent would achieve an accuracy higher than 99%. Nevertheless, that is the opposite of what we want. We do not want a 99% accuracy that is achieved by never labeling a transaction as fraudulent, we want to detect fraudulent transactions and label them as such.
To create our balanced training data set, I took all of the fraudulent transactions in our data set and counted them. Then, I randomly selected the same number of non-fraudulent transactions and concatenated the two. After shuffling this newly created data set, I decided to output the class distributions once more to visualize the difference.
<b> Note: </b> The sata set we created isn't completely balanced, The dataset contained 62.5% non-fraud transactions while 37.5% fraud transactions but it is good for making classification model. 

<br><br><br>
![Screenshot_64](https://user-images.githubusercontent.com/46135898/68549135-b9ba3000-0416-11ea-9e18-266f32799ce3.png)
<br><br><br>
![Screenshot_65](https://user-images.githubusercontent.com/46135898/68549137-baeb5d00-0416-11ea-9d1a-ba9cef328a2c.png)
<br><br><br>
![Screenshot_66](https://user-images.githubusercontent.com/46135898/68549138-bb83f380-0416-11ea-855a-fa32c831851c.png)
<br><br><br>
And then we created <b>train.csv</b> and <b>test.csv</b>.

[Go Back](#back)

<a id='h3'></a>
### 3. Data Visualization:
After that we visualize data distribution between both classes with the help of different charts. We'll also make charts to visualize Correlation of all features with target variable(Class).

<br><br><br>
![Screenshot_64](https://user-images.githubusercontent.com/46135898/68549135-b9ba3000-0416-11ea-9e18-266f32799ce3.png)
<br><br><br>
![Screenshot_65](https://user-images.githubusercontent.com/46135898/68549137-baeb5d00-0416-11ea-9d1a-ba9cef328a2c.png)
<br><br><br>
![Screenshot_66](https://user-images.githubusercontent.com/46135898/68549138-bb83f380-0416-11ea-855a-fa32c831851c.png)
<br><br><br>
![Screenshot_67](https://user-images.githubusercontent.com/46135898/68549139-bcb52080-0416-11ea-8627-5f4db15fb922.png)
<br><br><br>
![Screenshot_68](https://user-images.githubusercontent.com/46135898/68549140-bd4db700-0416-11ea-9388-b198328dceb5.png)
<br><br><br>

[Go Back](#back)

<a id='h4'></a>
### 4. Feature Normalization:
Feature normalization makes the values of each feature in the data have zero-mean (when subtracting the mean in the numerator) and unit-variance. In this dataset we have only one feature named <b>"normalizedAmount"</b> having values greater than 1, except this feature, all other have values in range 0. So we apply feature normalization on <b>"normalizedAmount"</b>.

<br><br><br>
![Screenshot_69](https://user-images.githubusercontent.com/46135898/68549141-be7ee400-0416-11ea-9d56-ea234432e185.png)
<br><br><br>

[Go Back](#back)

<a id='h5'></a>
### 5. Data Selection:
Now we make subset of features that have high impact on target variable(class).

<br><br><br>
![Screenshot_70](https://user-images.githubusercontent.com/46135898/68549142-bf177a80-0416-11ea-9883-344eea41b549.png)
<br><br><br>

[Go Back](#back)

<a id='h6'></a>
### 6. Model Selection:
From the past experience, we become to know that in Classification Algorithms, the best algorithms in terms of efficiency and accuracy are <b>Random Forest </b>and <b> XG Boost</b>, so we use both of them to make a model and then select the better one from both of them.

<a id='h6.1'></a>
#### 6.1. Random Forest Model:
Now we are able to make ML model for classification purpose. First of all, we have choose <b> Random Forest</b> model. It is best supervised learning algorithm. It is considered as a highly accurate and robust method because of the number of decision trees participating in the process. So the process of creating model and training and testing model is given below:

First we'll apply <b> Grid Search</b> on Random Forest to find best hyperparametrs.

<br><br><br>
![Screenshot_71](https://user-images.githubusercontent.com/46135898/68549143-c048a780-0416-11ea-87d0-17729076561e.png)
<br><br><br>

After finding best hyperparametrs, we'll apply cross validation on Random forsest to find <b> average accuray score, f1-score, roc-auc score and log-loss</b>.

<br><br><br>
![Screenshot_72](https://user-images.githubusercontent.com/46135898/68549144-c0e13e00-0416-11ea-95d6-9373f83b5809.png)
<br><br><br>

Then we extract features and make subset of best features by finding feature importance from random forest algo.

<br><br><br>
![Screenshot_73](https://user-images.githubusercontent.com/46135898/68549145-c2126b00-0416-11ea-8091-3424d496a37e.png)
<br><br><br>
![Screenshot_74](https://user-images.githubusercontent.com/46135898/68549146-c50d5b80-0416-11ea-83c5-5207375c5ee2.png)
<br><br><br>

But features that I have extracted from random Forest algo(X11-X15) weren't giving result as good as  the features we extracted before by correlation(X1-X10). So I selected those features for training and testing purpose.

<br><br><br>
![Screenshot_75](https://user-images.githubusercontent.com/46135898/68549147-c63e8880-0416-11ea-9e1c-650fa7574bbb.png)
<br><br><br>

Now I trained and tested the model on training and testing data. It has given the following result.

<br><br><br>
![Screenshot_76](https://user-images.githubusercontent.com/46135898/68549148-c6d71f00-0416-11ea-8eb9-dbc5c41040b5.png)
<br><br><br>

After that I finalized the best <b> Random Forest Model</b> that is given below:

<br><br><br>
![Screenshot_77](https://user-images.githubusercontent.com/46135898/68549150-c8084c00-0416-11ea-9e4d-e8f7043ba773.png)
<br><br><br>

#### Confusion Matrix of Best RF Model:

<br><br><br>
![Screenshot_78](https://user-images.githubusercontent.com/46135898/68549151-c9397900-0416-11ea-8ef3-c1de90852191.png)
<br><br><br>

[Go Back](#back)

#### 6.2. XGBoost:
XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. So after Random Forest, we created XGBoost model to train our data. It followed same process that was followed by Random Forest before.

First I applied <b> Grid Search</b> on XGB to find best hyperparametrs.

After that I tried to apply cross validation on XGB, but it was taking too much time. So first, I reduced number of a subset of features. To reduce the number of subset of features, first I extracted most relevant features from XGBoost algo.

<br><br><br>
![Screenshot_79](https://user-images.githubusercontent.com/46135898/68550239-e4aa8100-0422-11ea-930a-6014a5a897af.png)
<br><br><br>

The above process reduced the number of subset of features, and then the new new subset was X11-X16.

<br><br><br>
![Screenshot_80](https://user-images.githubusercontent.com/46135898/68550242-e70cdb00-0422-11ea-857e-4187f7127136.png)
<br><br><br>


Then I applied <b>Grid Search </b> on new subset of features and they all were giving the accuracy of 0.94. So I selected subset that have minimum number of features(they are X13-X16) and applied cross validation on them.
You can see the result of cross validation:

<br><br><br>
![Screenshot_81](https://user-images.githubusercontent.com/46135898/68550243-e83e0800-0422-11ea-9126-fdbf465a8e7e.png)
<br><br><br>


Then we selected XGB hyperparametrs with best result.

<br><br><br>
![Screenshot_82](https://user-images.githubusercontent.com/46135898/68550244-e96f3500-0422-11ea-9a0c-da657bb04065.png)
<br><br><br>


Now I trained and tested the model on training and testing data. It has given the following result.

<br><br><br>
![Screenshot_83](https://user-images.githubusercontent.com/46135898/68550245-ea07cb80-0422-11ea-800d-70e5d8c4fdc3.png)
<br><br><br>



I finalized 2 XGB models that were giving best result.

<br><br><br>
![Screenshot_84](https://user-images.githubusercontent.com/46135898/68550246-eaa06200-0422-11ea-8562-72645d49fd75.png)
<br><br><br>



#### Confusion Matrix of Best XGB Model:

<br><br><br>
![Screenshot_85](https://user-images.githubusercontent.com/46135898/68550247-ebd18f00-0422-11ea-83ed-76579c94cee4.png)
<br><br><br>
![Screenshot_86](https://user-images.githubusercontent.com/46135898/68550250-f1c77000-0422-11ea-85c6-d28fc472bee8.png)
<br><br><br>

### 7- Summerize Models with their results:
In the end, we have finalized 3 models that are giving best results, 1 related to random forest and other 2 belongs to XGBoost, they are given below.

<br><br><br>
![Screenshot_87](https://user-images.githubusercontent.com/46135898/68550549-8d59e000-0425-11ea-8954-fdc506cbf7df.png)
<br><br><br>
![Screenshot_88](https://user-images.githubusercontent.com/46135898/68550550-8e8b0d00-0425-11ea-8278-5353b0c36889.png)







