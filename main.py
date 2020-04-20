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
RANDOM_PREFIX=np.random.randint(1000,9999)

def main():
    #%% Reading data
    train=pd.read_csv('./data/cs-training.csv',index_col=0)
    X=train.drop('SeriousDlqin2yrs',axis=1)
    y=train.SeriousDlqin2yrs

    #Feature Engineering
    print(f'Starting shape: {X.shape}')
    X=(X
        .pipe(utils.replace_w_sensible_values)
        .pipe(utils.replace_na)
        .pipe(utils.log_transform_df)
        .pipe(utils.add_AgeDecade)
        .pipe(utils.add_boolean_DebtRatio_33)
        .pipe(utils.add_boolean_DebtRatio_43)
        .pipe(utils.add_features_per_dependent)
        .pipe(utils.add_features_per_creditline)
        .pipe(utils.add_features_per_estate)
        .pipe(utils.add_features_distance_from_mean)
        .pipe(utils.add_features_distance_from_median)
        .pipe(utils.add_features_distance_from_std)
    )
    print(f'Post feature engineering shape:{X.shape}')


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
        scale_pos_weight=13.960106382978724 #T/P-1
    )

    # %%
    skf=StratifiedKFold(
        n_splits=5,
        random_state=RANDOM_STATE
    )

    # %%
    tuning_params = {
        'num_leaves':[5,10,15,31,40,50],
        'scale_pos_weight':[1,10,14,16], # T/P-1 = 13.96
        'n_estimators':[100,250,500,750,1000],
        'learning_rate':[0.025,0.01,0.05,0.1],
        'max_depth':[3,5,7],
        'min_child_weight':range(3,6,1),
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)],
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
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

    # %% Extraction and selection of final features
    impt_features=utils.get_feature_importance(gs,X_train)
    impt_features.to_csv(f'output/{RANDOM_PREFIX}_impt_features.csv',index=False)
    final_features= impt_features[impt_features['importance']>0].feature.values
    X_train=X_train.loc[:, final_features]
    X_test=X_test.loc[:,final_features]
   
    # %% Final Model
    best_lgb=lgb.LGBMClassifier().set_params(**gs.best_params_)
    best_lgb.fit(X_train,y_train)

    # %% Model Performance, AUC and PRC curves
    utils.calculate_model_performance(
        best_lgb,
        X_train,
        X_test,
        y_train,
        y_test,
        RANDOM_PREFIX
    )

    # %% Create Submission
    utils.create_submission(best_lgb,final_features,RANDOM_PREFIX)
    
if __name__ == "__main__":
    main()
