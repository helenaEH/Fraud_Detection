# Fraud Detection

This project is about detecting fraudulent transactions from online payment data. The dataset is based on real company data and it is full of missing values and confidential/unknown features. This tutorial walks you through the process of building a data science project about fraud detection and the relevant things one should consider when dealing with anomaly detection problems in general. 
The competition host doesn't allow to share the data externally but it can be accessed on Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data

**Disclaimer: this project was built in a week and will not top the Kaggle leaderboard, however, it is a good example of a typical data project pipeline** 

## Part 1. Exploratory Data Analysis

EDA.py file has the python script to explore the data. Main findings: 
* The competition host provides 2 datasets: transaction and identity data
* Transaction data shape: rows 590540, columns 394; identity data: rows 144233, columns 41
* 41% of the data is missing from the transaction file and 35% from the identity file. 
* The identity file only includes 24% of the clients from the transaction data, rest of the information is missing 
* 3.5% of the transactions are fraudulent 
* Many features are categorical 

## Part 2. Data Wrangling 
The data has missing values and categorical variables. To prepare the data for the model we can do the following:
* Choose features that have less than 10% of missing data
* Use different functions on Vxx confidential features (nunique, sum)
* Factorize categorical features with 2 unique values
* One-hot encode other categorical non-numeric features
* addr1, addr2 are numerical categorical features that have too many unique values to use one-hot encoding (we want to avoid having too high dimensionality), use mean target encoding instead
* rest of the missing values fill with mean / mode depending on whether the variable is continuous or categorical

## Part 3. Train Test split, SMOTE, scaling
* Splitting the data into a train and test sets. 

* In anomaly detection the class imbalance is a typical problem. Use Synthetic Minority Oversampling Technique (SMOTE) to balance the classes by up-sampling the minority class. 
&nbsp; 
![](https://github.com/helenaEH/Fraud_Detection/blob/master/plots/class_imbalance.png)  

* Some features (e.g. Transaction Amount, TransactionDT) are on a larger scale than other variables, that will bias our model towards them. Normalize by scaling these features. 

## Part 4: Models and outcomes 
I tested 5 different models to compare their performance: 
* Logistic Regression 
* Linear Discriminant analysis 
* Random Forest 
* XGBoost 
* LightGB

The LightGB - a gradient boosting framework that uses tree-based learning algorithms performed the best. Simple Logistic Regression had the worst performance as expected. 

About the problem: we are testing our data on unbalanced test set, we don't want to overfit it. We would expect most of the predictions be classified under True Negatives - most of the data is non-fraudulent. Our goal is to keep the False Positive (non-fraudulent transactions mistakenly classifyed as fraudulent) and False Negatives (fraudulent transactions that our algorithm classified as safe) as low as possible. 

A few examples for illustration:

**Random Forest**

&nbsp; 
![](https://github.com/helenaEH/Fraud_Detection/blob/master/plots/confusion_matrix_Random%20Rorest.png)  

&nbsp; 
![](https://github.com/helenaEH/Fraud_Detection/blob/master/plots/roc_Random%20Rorest.png)  

**LightGB**

&nbsp; 
![](https://github.com/helenaEH/Fraud_Detection/blob/master/plots/confusion_matrix_LightGB.png)  

&nbsp; 
![](https://github.com/helenaEH/Fraud_Detection/blob/master/plots/roc_LightGB.png)  

### False Positive and False negative rate (the lower the better)

&nbsp; 
![](https://github.com/helenaEH/Fraud_Detection/blob/master/plots/Model_performance.png)  

### Overall model performance

&nbsp; 
![](https://github.com/helenaEH/Fraud_Detection/blob/master/model_outcome.PNG)  

## Future Improvements

* Ensemble modeling - combining different models with different weights (e.g. LightGB, Random Forest and XGBoost) to improve the overall accuracy 










 




