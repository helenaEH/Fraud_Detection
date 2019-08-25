""" Train different models, print and save confusion matrix, ROC curve, overall model performance metrics """

from data_wrangling import X, y
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

def prediction(model, xtest):
    predict = model.predict(xtest)
    return predict

def conf_matrix(ytest, ypred, model_name):
    plt.figure()
    cm = confusion_matrix(ytest, ypred)
    labels = ['Not Fraud', 'Fraud']
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', vmin=0.2, cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f'plots/confusion_matrix_{model_name}.png')
    plt.show()
    return cm

def evaluation(ypred, ytest, model):
    cm = confusion_matrix(ytest, ypred)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    false_pos_rate = FP / (TN + FP)
    false_neg_rate = FN / (TP + FN)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'False positive rate: {false_pos_rate}')
    print(f'False negative rate: {false_neg_rate}')
    f = open("models.txt", "a+")
    f.write(
        f'{model}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nFalse positive rate: '
        f'{false_pos_rate}\nFalse negative rate: {false_neg_rate}\n\n')
    return false_pos_rate, false_neg_rate

def plot_roc(model, model_name):
    fpr, tpr, threshold = roc_curve(y_test, model.predict(X_test))
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'plots/roc_{model_name}.png')
    return plt.show()

def logreg(xtrain, ytrain, xtest, ytest):
    """ Logistic regression """
    model_name = 'Logistic Regression'
    lr = LogisticRegression(random_state=12)
    lr.fit(xtrain, ytrain)
    pred = prediction(lr, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, model_name)
    plot_roc(lr, model_name)
    return lr, false_pos, false_neg

def lda(xtrain, ytrain, xtest, ytest):
    """Linear discriminant analysis"""
    model_name = 'Linear Discriminant Analysis'
    pca = PCA(n_components=20)
    xtrain_pca = pca.fit_transform(xtrain)
    m_lda = LinearDiscriminantAnalysis(n_components=15)
    m_lda.fit(xtrain_pca, ytrain)
    xtest_pca = pca.transform(xtest)
    pred = prediction(m_lda, xtest_pca)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, model_name)
    plot_roc(m_lda, model_name)
    return m_lda, false_pos, false_neg

def randforest(xtrain, ytrain, xtest, ytest):
    """ Random Forest Classifier"""
    model_name = 'Random Rorest'
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=12, max_depth=100)
    clf_rf.fit(xtrain, ytrain)
    pred = prediction(clf_rf, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, 'Random Rorest')
    plot_roc(clf_rf, model_name)
    return clf_rf, false_pos, false_neg

def xgboost(xtrain, ytrain, xtest, ytest):
    """ XGBoost """
    model_name = 'XGBoost'
    xgb = XGBClassifier(max_depth=5, n_estimators=500, scale_pos_weight=4, random_state=12)
    xgb.fit(xtrain, ytrain)
    pred = prediction(xgb, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, model_name)
    plot_roc(xgb, model_name)
    return xgb, false_pos, false_neg

def lightgb(xtrain, ytrain, xtest, ytest):
    """ Light Gradient Boosting """
    model_name = 'LightGB'
    gbm = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.06, class_weight={0: 1, 1: 7})
    gbm.fit(xtrain, ytrain, eval_set=(xtest, ytest))
    pred = prediction(gbm, xtest)
    conf_matrix(ytest, pred, model_name)
    false_pos, false_neg = evaluation(pred, ytest, model_name)
    plot_roc(gbm, model_name)
    return gbm, false_pos, false_neg

def model_performance(fp, fn):
    objects = ('LogReg', 'LDA', 'RandForest', 'XGB', 'LightGB')
    y_pos = np.arange(len(objects))
    ax = plt.subplot(111)
    ax.bar(y_pos-0.2, fp, width=0.2, color='b', align='center')
    ax.bar(y_pos, fn, width=0.2, color='g', align='center')
    plt.xticks(y_pos, objects)
    plt.ylabel('Rate %')
    plt.title(f'Model Performance')
    plt.legend(['FP', 'FN'])
    plt.savefig('plots/Model_performance.png', bbox_inches='tight')
    return plt.show()

# Synthetic minority oversampling technique, outputs numpy array
sm = SMOTE(sampling_strategy=0.1, random_state=12)
X_train, y_train = sm.fit_sample(X_train, y_train)

# Converting numpy to pandas dataframe
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Normalize features with high values
scaler = StandardScaler()
X_train[['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5']] = scaler. \
    fit_transform(X_train[['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5']])

X_test[['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5']] = \
    scaler.transform(X_test[['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5']])

m_logreg, fp_logreg, fn_logreg = logreg(X_train, y_train, X_test, y_test)
lda, fp_lda, fn_lda = lda(X_train, y_train, X_test, y_test)
rand_forest, fp_rf, fn_rf = randforest(X_train, y_train, X_test, y_test)
xgboost, fp_xgb, fn_xgb = xgboost(X_train, y_train, X_test, y_test)
lightgb, fp_lgb, fn_lgb = lightgb(X_train, y_train, X_test, y_test)

false_positives = list([fp_logreg, fp_lda, fp_rf, fp_xgb, fp_lgb])
false_negatives = list([fn_logreg, fn_lda, fn_rf, fn_xgb, fn_lgb])
fp_fn = model_performance(false_positives, false_negatives)
