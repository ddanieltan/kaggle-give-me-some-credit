#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %% Ideas to improve Kaggle leaderboard position
# We start from position 663
# 1. Lgb Model parameter tuning
# a. scale_pos_weight to better encode unbalanced classes
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53696
# b. gridsearch/randomsearch

# 2. More Feature engineering -> bring in domain knowledge

#%% Globals
RANDOM_STATE = 2020

#%% Data prep from 2-model.py
train=pd.read_csv('./data/cs-training.csv',index_col=0)
X=train.drop('SeriousDlqin2yrs',axis=1)
y=train.SeriousDlqin2yrs

#%% New Features
# Bin age by decade
X['AgeDecade']=pd.cut(x=X['age'], bins=[20,29, 39,49,59,69,79,89,99,109], labels=[20,30,40,50,60,70,80,90,100])

#%% Bin debt ratio
X['DebtRatio']=np.where(X['DebtRatio'] > 1.0, 1.0, X['DebtRatio'])
X['DebtRatio']=pd.cut(x=X['DebtRatio'], bins=[-0.1,0.33,0.43,1.0], labels=[0,1,2])


#%%
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

X.MonthlyIncome=X.MonthlyIncome.fillna(X.MonthlyIncome.median())
X.MonthlyIncome=np.log(X.MonthlyIncome+1)

X.NumberOfDependents=X.NumberOfDependents.fillna(X.NumberOfDependents.median())
X.NumberOfDependents=np.log(X.NumberOfDependents+1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=RANDOM_STATE,
    stratify=y
)




#%% Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)



#%% Retrieve current model
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

#%% Tuning parameters
tuning_params = {
    'num_leaves':[5,10,15,31,40,50],
    'scale_pos_weight':[1,10,14,16], # T/P-1 = 13.96
    'n_estimators':[100,250,500,750,1000],
    'learning_rate':[0.025,0.01,0.05,0.1],
    'max_depth':[2,4,5,10,20]
}
#%%
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(
    n_splits=3,
    random_state=RANDOM_STATE
)
from sklearn.model_selection import RandomizedSearchCV

gs=RandomizedSearchCV(
    estimator=lgb_model, 
    param_distributions=tuning_params, 
    n_iter=50,
    scoring='roc_auc',
    cv=skf,
    refit=True,
    verbose=True)

# %% Fitting Randomsearch
gs.fit(X_train_scaled,y_train)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
#Best score reached: 0.8637597836736455 with params: {'scale_pos_weight': 10, 'num_leaves': 10} 

# %%
lgb_model_improved=lgb.LGBMClassifier(
    n_estimators=200, 
    silent=False, 
    random_state=RANDOM_STATE, 
    max_depth=4,
    objective='binary',
    metrics ='auc',
    boosting='gbdt',
    learning_rate=0.025,
    scale_pos_weight=10,
    num_leaves=10
)
lgb_model_improved.fit(X_train_scaled,y_train)

# %%
def create_submission(model):
    kaggle_test=pd.read_csv('./data/cs-test.csv',index_col=0)
    submission=pd.read_csv('./data/sampleEntry.csv',index_col=0)
    X_kaggle=kaggle_test.drop('SeriousDlqin2yrs',axis=1)
    X_kaggle=scaler.transform(X_kaggle)
    y_pred=model.predict_proba(X_kaggle)[:,1]
    submission.Probability=y_pred

    #Save as csv
    prefix=str(np.random.randint(1000,9999))
    filepath=f'./output/{prefix}_submission.csv'
    submission.to_csv(filepath)
    print(f'Saved to {filepath}')


# %%
create_submission(lgb_model_improved)

# %%
