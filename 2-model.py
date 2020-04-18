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

#%% For features with missing values
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

# Scaling numeric features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

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
logreg.fit(X_train_scaled,y_train)
y_pred=logreg.predict_proba(X_test_scaled)[:,1]


#%% AUC scoring
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,f1_score
import matplotlib.pyplot as plt

def plot_roc_curve(y_true,y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")

def auc_plot_and_score(y_true,y_pred):
    plot_roc_curve(y_true,y_pred)
    print(roc_auc_score(y_test,y_pred)
)

#%%
auc_plot_and_score(y_test,y_pred)
# 0.83087

#%% Model 2: Random Forests


#%% RF Model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=500,
    max_depth=5
)
rf.fit(X_train_scaled,y_train)
y_pred=rf.predict_proba(X_test)[:,1]

#%%
auc_plot_and_score(y_test,y_pred)
# 0.81599

#%% Inspect feature importance with RF model
def plot_feature_importances(model):
    plt.figure(figsize=(10,8))
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(rf)
#NumberOfTimes90DaysLate highest importance


#%% Model 3: LightGBM
# No need to scale features, as this step was previously done for RF model
import lightgbm as lgb

#%% Parameters of LightGBM model
lgb_model=lgb.LGBMClassifier(
    n_estimators=200, 
    silent=False, 
    random_state=RANDOM_STATE, 
    max_depth=4,
    objective='binary',
    metrics ='auc',
    boosting='gbdt',
    learning_rate=0.025

)
lgb_model.fit(X_train_scaled,y_train)
y_pred=lgb_model.predict_proba(X_test)[:,1]

#%% LGB score
auc_plot_and_score(y_test,y_pred) #0.83199

#%% Use Cross Validation to score the 3 models, controlling for overfitting
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(
    n_splits=10,
    random_state=RANDOM_STATE
)

#%%
from sklearn.model_selection import cross_val_score

def cross_validated_auc(model,skf):
    scores=cross_val_score(
        model,
        X,
        y,
        scoring='roc_auc',
        cv=skf
    )

    print(f'Mean AUC: {np.mean(scores):.3f}, Std Dev: {np.std(scores):.3f}')


# %%
cross_validated_auc(logreg,skf)
cross_validated_auc(rf,skf)
cross_validated_auc(lgb_model,skf)

#
# Mean AUC: 0.8331686429732408, Std Dev: 0.0059887103906922975
# Mean AUC: 0.8590119994368026, Std Dev: 0.0045709444241961595
# Mean AUC: 0.8637209256175506, Std Dev: 0.0047222826253001315

#%% Submission
kaggle_test=pd.read_csv('./data/cs-test.csv',index_col=0)
submission=pd.read_csv('./data/sampleEntry.csv',index_col=0)
# %%
X_kaggle=kaggle_test.drop('SeriousDlqin2yrs',axis=1)
X_kaggle=scaler.transform(X_kaggle)
=
# %%
y_pred=lgb_model.predict_proba(X_kaggle)[:,1]

# %%
submission.Probability=y_pred

# %%
import random
prefix=str(random.randint(1000,9999))
submission.to_csv(f'./output/{prefix}_submission.csv')

# %%
