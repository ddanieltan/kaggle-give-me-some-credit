#%%
import utils
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

#%% Globals
RANDOM_STATE = 2020

#%% 
train=pd.read_csv('./data/cs-training.csv',index_col=0)
X=train.drop('SeriousDlqin2yrs',axis=1)
y=train.SeriousDlqin2yrs

#Feature Engineering


X=(X
    .pipe(utils.replace_w_sensible_values)
    .pipe(utils.replace_na)
    .pipe(utils.log_transform_df)
    .pipe(utils.add_AgeDecade)
    # .pipe(utils.add_DebtRatioBin)
    .pipe(utils.add_features_per_dependent)
    .pipe(utils.add_features_distance_from_mean)
    .pipe(utils.add_features_distance_from_median)
    .pipe(utils.add_features_distance_from_std)
)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=RANDOM_STATE,
    stratify=y
)

# %%
lgb_model=lgb.LGBMClassifier(
    silent=False, 
    random_state=RANDOM_STATE, 
    objective='binary',
    metrics ='auc',
    boosting='gbdt',
    scale_pos_weight=14
)

# %%
skf=StratifiedKFold(
    n_splits=10,
    random_state=RANDOM_STATE
)

# %%
tuning_params = {
    'num_leaves':[5,10,15,31,40,50],
    'scale_pos_weight':[1,10,14,16], # T/P-1 = 13.96
    'n_estimators':[100,250,500,750,1000],
    'learning_rate':[0.025,0.01,0.05,0.1],
    'max_depth':[3,5,7]
}
gs=RandomizedSearchCV(
    estimator=lgb_model, 
    param_distributions=tuning_params, 
    n_iter=50,
    scoring='roc_auc',
    cv=skf,
    refit=True,
    verbose=True)

gs.fit(X_train,y_train)

#%%
# Best score reached: 0.8651849275748571 with params: {'scale_pos_weight': 14, 'num_leaves': 10, 'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.01} 

# %% Submission
utils.create_submission(gs)


# %%
