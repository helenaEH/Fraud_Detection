""" Script that prints out results from the data exploration """

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()

# read csv files into pandas dataframes
train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')

# size of the dataframes
print('Transaction and identity dataframe shapes')
print(train_transaction.shape)
print(train_identity.shape)

# split transaction file into 4, because it has too many columns and print out overview of all the features
print('\nTransaction data features')
print(train_transaction.iloc[:, :100].info())
print(train_transaction.iloc[:, 100:200].info())
print(train_transaction.iloc[:, 200:300].info())
print(train_transaction.iloc[:, 300:].info())
print('\nIdentity data features')
print(train_identity.info())

# columns with no missing values
print('\nColumns with no values missing')
print(train_transaction.columns[train_transaction.notnull().all()])
print(train_identity.columns[train_identity.notnull().all()])

# % of missing values in the identity data
print('\n % of missing values in the identity dataframe')
print(f'{round(train_identity.isnull().sum().sum()*100/(len(train_identity)*len(train_identity.columns)), 2)} %')

# % of missing values in the transaction data
print(' \n% of missing values in the transaction dataframe\n')
print(
    f'{round(train_transaction.isnull().sum().sum()*100/(len(train_transaction)*len(train_transaction.columns)), 2)} %')

# Categorical data in the dataframes
transaction_cols = train_transaction.columns
transaction_num_cols = train_transaction.select_dtypes([np.number]).columns
transaction_categ = list(set(transaction_cols) - set(transaction_num_cols))
print('\nNon-numeric categorical features in the transaction dataset')
print(transaction_categ)

identity_cols = train_identity.columns
identity_num_cols = train_identity.select_dtypes([np.number]).columns
identity_categ = list(set(identity_cols) - set(identity_num_cols))
print('\nNon-numeric categorical features in the identity dataset')
print(identity_categ)

# check if all unique identity customers are included in the transaction dataset
transaction_ID = set(train_transaction.TransactionID)
identity_ID = set(train_identity.TransactionID)
print('\nAll identity customers are in the transaction dataset')
print(len(transaction_ID.intersection(identity_ID)))

# plotting fraudulent vs non-fraudulent transactions -> class imbalance
objects = ('Fraud', 'NotFraud')
y_pos = np.arange(len(objects))
fraud = len(train_transaction[train_transaction.isFraud == 1])
not_fraud = len(train_transaction[train_transaction.isFraud == 0])
total_fraud = [fraud, not_fraud]

plt.figure(figsize=(4, 3))
plt.bar(y_pos, total_fraud, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Total count')
plt.title(f'{round(fraud/(fraud+not_fraud)*100, 2)}% of transactions are fraudulent')
plt.savefig('plots/fraud_vs_notfraud.png', bbox_inches='tight')
plt.show()

# demonstrate class imbalance for the transaction amount data
x_plot = train_transaction.iloc[:1000, 3]
y_plot = train_transaction.index[:1000]
label = train_transaction.iloc[:1000, 1]
colors = ['steelblue', 'red']

# demonstrate class imbalance on a scatterplot
fig = plt.figure(figsize=(5, 4))
plt.scatter(x_plot, y_plot, c=label, cmap=ListedColormap(colors))
cb = plt.colorbar()
loc = np.arange(0, max(label), max(label)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(['Not Fraud', 'Fraud'])
plt.title('Class Imbalance')
plt.xlabel('Purchase amount')
plt.ylabel('CustomerID')
plt.savefig('plots/class_imbalance.png', bbox_inches='tight')
plt.show(fig)
