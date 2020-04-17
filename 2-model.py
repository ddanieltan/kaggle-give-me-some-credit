#%%
import pandas as pd
import numpy as np

#%% Globals
RANDOM_STATE = 2020

#%% Reading data
train=pd.read_csv('./data/cs-training.csv',index_col=0)
X=train.drop('SeriousDlqin2yrs',axis=1)
y=train.SeriousDlqin2yrs

#%% Feature Engineering

#%% Log Transform Skewed Features
skewed_features=[
    'RevolvingUtilizationOfUnsecuredLines',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse'
]

for feature_name in skewed_features:
    X[feature_name]=np.log(train[feature_name]+1)

#%% For features with issing values
# Replace with median, then log transform
X.MonthlyIncome=X.MonthlyIncome.fillna(X.MonthlyIncome.median())
X.MonthlyIncome=np.log(X.MonthlyIncome+1)

X.NumberOfDependents=X.NumberOfDependents.fillna(X.NumberOfDependents.median())
X.NumberOfDependents=np.log(X.NumberOfDependents+1)


#%% Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=RANDOM_STATE,
    stratify=y
)

#%% Verifying stratified sample
def check_proportion_of_classes(df):
    vc=df.value_counts()
    print(vc)
    print(f'Proportion of class 1: {vc[1]/vc.sum()}')

check_proportion_of_classes(y_train)
check_proportion_of_classes(y_test)
# Indeed, proportion of 0.066 is preserved in both train and test set

#%% Model 1: Logreg
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(
    random_state=RANDOM_STATE,
    penalty="l2",
    C=1
)
logreg.fit(X_train,y_train)
y_pred=logreg.predict_proba(X_test)[:,1]


#%% AUC scoring
from sklearn.metrics import roc_curve, roc_auc_score 

def plot_roc_curve(y_true,y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    plt.figure(figsize=(12,10))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")

plot_roc_curve(y_test,y_pred)

#%%
roc_auc_score(y_test,y_pred)

#%% Cross Validation
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(
    n_splits=10,
    random_state=RANDOM_STATE
)
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

