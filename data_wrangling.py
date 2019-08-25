"""
Data wrangling function that removes columns with too many nans, use aggregation functions for Vxxx features,
fills nans with mean or mode depending on whether the data is continuous or numeric categorical, factorizes
or uses one-hot encoding for categorical features, mean target encoding for address features
"""

import pandas as pd
import numpy as np
import category_encoders as ce
from constants import FINAL_COLS

train_transaction = pd.read_csv('train_transaction.csv')


def feature_engineering(dataframe):
    """ Input full data, fills, aggregates, encodes values, outputs fully cleaned dataframe """
    # mean target encoding
    mean_target = dataframe[['P_emaildomain', 'addr1', 'addr2']]
    emaildomain_enc = enc.transform(mean_target)
    emaildomain_enc.columns = ['P_emaildomain_1', 'addr1_1', 'addr2_1']
    dataframe = pd.DataFrame.join(dataframe, emaildomain_enc, how='outer')

    # finds if purchaser and receiver email domains are the same
    dataframe['rec_pur_domain'] = np.where(dataframe['R_emaildomain'] == dataframe['P_emaildomain'], 1, 0)

    # fill card values - card 3 has few unique values use mode instead of mean as it is not continuous
    dataframe['card2'].fillna(dataframe['card2'].mean(), inplace=True)
    dataframe['card3'].fillna(dataframe['card3'].mode()[0], inplace=True)
    dataframe['card5'].fillna(dataframe['card5'].mean(), inplace=True)

    # fill categorical feature missing values with 'unknown'
    categ_feat = ['card4', 'card6']
    for i in categ_feat:
        dataframe[i].fillna('unknown', inplace=True)
    dataframe['card6'].replace(['debit or credit', 'charge card'], 'unknown', inplace=True)

    # Factorizes columns according to input letter, add 3 for missing data
    m_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
    d = {'T': 1, 'F': 0}
    for i in m_cols:
        dataframe[i] = dataframe[i].map(d)
    for j in m_cols:
        dataframe[j].fillna(3, inplace=True)

    # Code that creates list of lists of batches of 11 items
    def batch(iterable, n=1):
        iter_len = len(iterable)
        for ndx in range(1, iter_len, n):
            yield iterable[ndx:min(ndx + n, iter_len)]

    list_of_lists = []
    for x in batch(range(0, 339), 11):
        list_of_lists.append(list(x))

    # code that sums up batches of 11 Vxxx columns for each row
    sum_list = []
    df_v = dataframe.loc[:, 'V1':'V339']

    for i in list_of_lists:
        sum_10 = np.array(df_v.iloc[:, i].sum(axis=1))
        sum_list.append(sum_10)
    v_sum = pd.DataFrame(np.row_stack(sum_list)).transpose()

    # Counts unique Vxxx feature values
    unique_count = df_v.nunique(axis=1)

    # Use one hot encoding for categrical non-numeric data
    encode_feat = ['card4', 'card6', 'ProductCD', 'M4']
    dataframe = pd.DataFrame.join(dataframe, pd.get_dummies(dataframe[encode_feat]), how='outer')

    # Fill these columns with mode because they are not continuous
    categoric_numeric = ['dist1', 'dist2']
    for i in categoric_numeric:
        dataframe[i].fillna(dataframe[i].mode()[0], inplace=True)

    # Drop columns that are engineered to new features
    dataframe.drop(labels=encode_feat, axis=1, inplace=True)
    dataframe.drop(labels=df_v.columns, axis=1, inplace=True)
    dataframe.drop(labels=['R_emaildomain', 'P_emaildomain', 'addr1', 'addr2'], axis=1, inplace=True)

    # Append engineered features with the dataframe
    pd.DataFrame.join(dataframe, v_sum, how='outer')
    dataframe['unique_count'] = unique_count

    dataframe.set_index('TransactionID', inplace=True)
    columns = FINAL_COLS
    dataframe = dataframe[columns]

    # Fill rest with averages of columns
    num_cols = dataframe.select_dtypes([np.number]).columns
    for i in list(num_cols):
        dataframe[i].fillna(dataframe[i].mean(), inplace=True)

    return dataframe


train_transaction['addr1'] = train_transaction['addr1'].apply(str)
train_transaction['addr2'] = train_transaction['addr2'].apply(str)

y = train_transaction['isFraud']
X_transaction = train_transaction.drop('isFraud', axis=1)
enc = ce.TargetEncoder()
enc.fit(train_transaction[['P_emaildomain', 'addr1', 'addr2']], y)
X = feature_engineering(X_transaction)
