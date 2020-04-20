import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Feature Engineering Functions
def replace_w_sensible_values(df_):
    '''
    Input: dataframe
    Output: dataframe with replaced values
    '''
    df=df_.copy()
    # Debt Ratio can't be >1
    df['DebtRatio']=np.where(df['DebtRatio'] > 1.0, 1.0, df['DebtRatio'])
    
    # Age can't be 0
    df.loc[df['age'] == 0, 'age'] = 21

    return df

def replace_na(df_):
    '''
    Input: dataframe
    Output: dataframe without na values
    '''
    df=df_.copy()

    # Assume minimum income of 1
    df['MonthlyIncome']=df['MonthlyIncome'].fillna(1.0)
    
    # Consider one's self a dependent
    df['NumberOfDependents']=df['NumberOfDependents']+1

    return df

def log_transform_df(df_):
    '''
    Input: dataframe, list of skewed features
    Output: dataframe with logtranformed features 
    '''

    skewed_features=[
        'RevolvingUtilizationOfUnsecuredLines',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'MonthlyIncome',
        'NumberOfDependents'
    ]

    df=df_.copy()
    for feature in skewed_features:
        df[feature]=np.log(df[feature]+0.0000001)
    return df

def add_AgeDecade(df_):
    '''
    Input: dataframe
    Output: dataframe with feature binning age
    '''
    df=df_.copy()
    df['AgeDecade']=pd.cut(
        x=df['age'], 
        bins=[20,29,39,49,59,69,79,89,99,109],
        labels=[20,30,40,50,60,70,80,90,100]
    )
    return df



def add_boolean_DebtRatio_33(df_):
    '''
    Input: dataframe
    Output: dataframe with 1 added feature
    Read domain knowledge that Debt Ratio of 0.33 is commonly used threshold
    '''
    
    df=df_.copy()
    df['DebtRatio33']=df['DebtRatio']>0.33
    return df

def add_boolean_DebtRatio_43(df_):
    '''
    Input: dataframe
    Output: dataframe with 1 added feature
    Read domain knowledge that Debt Ratio of 0.43 is commonly used threshold
    '''
    
    df=df_.copy()
    df['DebtRatio33']=df['DebtRatio']>0.43
    return df


def add_features_per_dependent(df_):
    '''
    Input: dataframe
    Output: dataframe with additional 5 transformed features
    '''
    df=df_.copy()
    df['MonthlyIncomePerDependent']=df['MonthlyIncome']/df['NumberOfDependents']
    df['NumberRealEstateLoansOrLinesPerDependent']=df['NumberRealEstateLoansOrLines']/df['NumberOfDependents']
    df['NumberOfOpenCreditLinesAndLoansPerDependent']=df['NumberOfOpenCreditLinesAndLoans']/df['NumberOfDependents']
    df['NumberOfTime30-59DaysPastDueNotWorsePerDependent']=df['NumberOfTime30-59DaysPastDueNotWorse']/df['NumberOfDependents']
    df['NumberOfTime60-89DaysPastDueNotWorsePerDependent']=df['NumberOfTime60-89DaysPastDueNotWorse']/df['NumberOfDependents']
    return df

def add_features_per_creditline(df_):
    '''
    Input: dataframe
    Output: dataframe with additional 4 transformed features
    '''
    df=df_.copy()
    df['MonthlyIncomePerCreditLine']=df['MonthlyIncome']/df['NumberOfOpenCreditLinesAndLoans']
    df['NumberRealEstateLoansOrLinesPerCreditLine']=df['NumberRealEstateLoansOrLines']/df['NumberOfOpenCreditLinesAndLoans']
    df['NumberOfTime30-59DaysPastDueNotWorsePerCreditLine']=df['NumberOfTime30-59DaysPastDueNotWorse']/df['NumberOfOpenCreditLinesAndLoans']
    df['NumberOfTime60-89DaysPastDueNotWorsePerCreditLine']=df['NumberOfTime60-89DaysPastDueNotWorse']/df['NumberOfOpenCreditLinesAndLoans']

    return df

def add_features_per_estate(df_):
    '''
    Input: dataframe
    Output: dataframe with additional 4 transformed features
    '''
    df=df_.copy()
    df['MonthlyIncomePerEstate']=df['MonthlyIncome']/df['NumberRealEstateLoansOrLines']
    df['NumberRealEstateLoansOrLinesPerEstate']=df['NumberRealEstateLoansOrLines']/df['NumberRealEstateLoansOrLines']
    df['NumberOfTime30-59DaysPastDueNotWorsePerEstate']=df['NumberOfTime30-59DaysPastDueNotWorse']/df['NumberRealEstateLoansOrLines']
    df['NumberOfTime60-89DaysPastDueNotWorsePerEstate']=df['NumberOfTime60-89DaysPastDueNotWorse']/df['NumberRealEstateLoansOrLines']

    return df

def add_features_distance_from_mean(df_):
    '''
    Input: dataframe
    Output: dataframe with additional 10 transformed features
    '''
    df=df_.copy()
    df['RevolvingUtilizationOfUnsecuredLinesMeanDist']=np.mean(df['RevolvingUtilizationOfUnsecuredLines'])-df['RevolvingUtilizationOfUnsecuredLines']
    df['ageMeanDist']=np.mean(df['age'])-df['age']
    df['NumberOfTime30-59DaysPastDueNotWorseMeanDist']=np.mean(df['NumberOfTime30-59DaysPastDueNotWorse'])-df['NumberOfTime30-59DaysPastDueNotWorse']
    df['DebtRatioDist']=np.mean(df['DebtRatio'])-df['DebtRatio']
    df['MonthlyIncomeDist']=np.mean(df['MonthlyIncome'])-df['MonthlyIncome']
    df['NumberOfOpenCreditLinesAndLoansMeanDist']=np.mean(df['NumberOfOpenCreditLinesAndLoans'])-df['NumberOfOpenCreditLinesAndLoans']
    df['NumberOfTimes90DaysLateMeanDist']=np.mean(df['NumberOfTimes90DaysLate'])-df['NumberOfTimes90DaysLate']
    df['NumberRealEstateLoansOrLinesMeanDist']=np.mean(df['NumberRealEstateLoansOrLines'])-df['NumberRealEstateLoansOrLines']
    df['NumberOfTime60-89DaysPastDueNotWorseMeanDist']=np.mean(df['NumberOfTime60-89DaysPastDueNotWorse'])-df['NumberOfTime60-89DaysPastDueNotWorse']
    df['NumberOfDependentsMeanDist']=np.mean(df['NumberOfDependents'])-df['NumberOfDependents']
    return df

def add_features_distance_from_median(df_):
    '''
    Input: dataframe
    Output: dataframe with additional 10 transformed features
    '''
    df=df_.copy()
    df['RevolvingUtilizationOfUnsecuredLinesMeanDist']=np.median(df['RevolvingUtilizationOfUnsecuredLines'])-df['RevolvingUtilizationOfUnsecuredLines']
    df['ageMeanDist']=np.median(df['age'])-df['age']
    df['NumberOfTime30-59DaysPastDueNotWorseMeanDist']=np.median(df['NumberOfTime30-59DaysPastDueNotWorse'])-df['NumberOfTime30-59DaysPastDueNotWorse']
    df['DebtRatioDist']=np.median(df['DebtRatio'])-df['DebtRatio']
    df['MonthlyIncomeDist']=np.median(df['MonthlyIncome'])-df['MonthlyIncome']
    df['NumberOfOpenCreditLinesAndLoansMeanDist']=np.median(df['NumberOfOpenCreditLinesAndLoans'])-df['NumberOfOpenCreditLinesAndLoans']
    df['NumberOfTimes90DaysLateMeanDist']=np.median(df['NumberOfTimes90DaysLate'])-df['NumberOfTimes90DaysLate']
    df['NumberRealEstateLoansOrLinesMeanDist']=np.median(df['NumberRealEstateLoansOrLines'])-df['NumberRealEstateLoansOrLines']
    df['NumberOfTime60-89DaysPastDueNotWorseMeanDist']=np.median(df['NumberOfTime60-89DaysPastDueNotWorse'])-df['NumberOfTime60-89DaysPastDueNotWorse']
    df['NumberOfDependentsMeanDist']=np.median(df['NumberOfDependents'])-df['NumberOfDependents']
    return df

def add_features_distance_from_std(df_):
    '''
    Input: dataframe
    Output: dataframe with additional 10 transformed features
    '''
    df=df_.copy()
    df['RevolvingUtilizationOfUnsecuredLinesMeanDist']=np.std(df['RevolvingUtilizationOfUnsecuredLines'])-df['RevolvingUtilizationOfUnsecuredLines']
    df['ageMeanDist']=np.std(df['age'])-df['age']
    df['NumberOfTime30-59DaysPastDueNotWorseMeanDist']=np.std(df['NumberOfTime30-59DaysPastDueNotWorse'])-df['NumberOfTime30-59DaysPastDueNotWorse']
    df['DebtRatioDist']=np.std(df['DebtRatio'])-df['DebtRatio']
    df['MonthlyIncomeDist']=np.std(df['MonthlyIncome'])-df['MonthlyIncome']
    df['NumberOfOpenCreditLinesAndLoansMeanDist']=np.std(df['NumberOfOpenCreditLinesAndLoans'])-df['NumberOfOpenCreditLinesAndLoans']
    df['NumberOfTimes90DaysLateMeanDist']=np.std(df['NumberOfTimes90DaysLate'])-df['NumberOfTimes90DaysLate']
    df['NumberRealEstateLoansOrLinesMeanDist']=np.std(df['NumberRealEstateLoansOrLines'])-df['NumberRealEstateLoansOrLines']
    df['NumberOfTime60-89DaysPastDueNotWorseMeanDist']=np.std(df['NumberOfTime60-89DaysPastDueNotWorse'])-df['NumberOfTime60-89DaysPastDueNotWorse']
    df['NumberOfDependentsMeanDist']=np.std(df['NumberOfDependents'])-df['NumberOfDependents']
    return df

# Evaluation functions
def plot_corr_heatmap(df):
    corr=df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))   
    plt.subplots(figsize=(20,15))
    sns.set(font_scale=5)
    chart=sns.heatmap(train.corr(),cmap="PiYG",mask=mask)
    chart.set_xticklabels([])
    return chart

def check_proportion_of_classes(ser):
    '''
    Input: Series
    Output: None

    Prints proportion of class 1
    '''
    vc=df.value_counts()
    print(vc)
    print(f'Proportion of class 1: {vc[1]/vc.sum()}')


def get_feature_importance(model,X_train):
    feat_df=pd.DataFrame(
        model.best_estimator_.feature_importances_,
        index=X_train.columns,
        columns=['importance']
    ).sort_values('importance',ascending=False)

    feat_df['normalized_importance']=feat_df['importance']/(
        feat_df['importance'].sum()+np.finfo(float).eps
    )

    feat_df['cumulative_importance']=feat_df['normalized_importance'].sort_values(ascending=False).cumsum()
    feat_df.reset_index(inplace=True)
    feat_df.rename(
        columns={'index':'feature'},
        inplace=True
    )

    return feat_df

def evaluation_metrics(y, y_pred):
    """
    Compute score statistics
    :param y: series, data label
    :param y_pred: series / ndarray, classifier score
    :return
        performance: dict, performance statistics
        misc: dict, raw data for performance
    """

    y_pred_cls = (y_pred > 0.5).astype(int)
    cr = classification_report(y, y_pred_cls, digits=4, output_dict=True)
    cm = confusion_matrix(y, y_pred_cls)

    fpr, tpr, _ = roc_curve(y, y_pred)
    auc = roc_auc_score(y, y_pred)

    precision, recall, _ = precision_recall_curve(y, y_pred)
    ap = average_precision_score(y, y_pred)

    performance = {
        'AUC': auc, 'AP': ap, 'F1': cr['1']['f1-score'],
        'PRECISION': cr['1']['precision'], 'RECALL': cr['1']['recall']
    }

    misc = {
        'classification_report': cr, 'confusion_matrix': cm,
        'fpr': fpr, 'precision': precision, 'recall': recall, 'tpr': tpr
    }

    return performance, misc

def calculate_model_performance(model, X_trainval, X_test, y_trainval, y_test, prefix):
    """
    Calculate model performance
    :param model: model object implementing `predict` and `predict_proba`
    :param X_trainval: DataFrame, trainval dataset
    :param X_test: DataFrame, test dataset
    :param y_trainval: Series, trainval label
    :param y_test: Series, test label
    """

    y_trainval_pred = model.predict_proba(X_trainval)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    pfm_trainval, misc_trainval = evaluation_metrics(y_trainval, y_trainval_pred)
    pfm_test, misc_test = evaluation_metrics(y_test, y_test_pred)

    # Print confusion matrix
    print('\nConfusion Matrix TrainVal:')
    print(misc_trainval['confusion_matrix'])
    print('\nConfusion Matrix Test:')
    print(misc_test['confusion_matrix'])

    print('\nClassification Report TrainVal:')
    print(pd.DataFrame(misc_trainval['classification_report']))
    print('\nClassification Report Test:')
    print(pd.DataFrame(misc_test['classification_report']))

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(misc_trainval['fpr'], misc_trainval['tpr'], label=f'TrainVal, AUC = {pfm_trainval["AUC"]:.3f}')
    plt.plot(misc_test['fpr'], misc_test['tpr'], label=f'Test, AUC = {pfm_test["AUC"]:.3f}')
    plt.legend(loc='lower right', prop={'size': 10})
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    plt.savefig(f'output/{prefix}_roc.png')

    # Plot PRC curve
    plt.figure(figsize=(8, 6))
    plt.plot(misc_trainval['recall'], misc_trainval['precision'], label=f'TrainVal, AP = {pfm_trainval["AP"]:.3f}')
    plt.plot(misc_test['recall'], misc_test['precision'], label=f'Test, AP = {pfm_test["AP"]:.3f}')
    plt.legend(loc='upper right', prop={'size': 10})
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRC Curve')
    plt.show()
    plt.savefig(f'output/{prefix}_prc.png')

# Submission function

def create_submission(model,prefix):
    kaggle_test=pd.read_csv('./data/cs-test.csv',index_col=0)
    submission=pd.read_csv('./data/sampleEntry.csv',index_col=0)
    X_kaggle=kaggle_test.drop('SeriousDlqin2yrs',axis=1)
    X_kaggle=(X_kaggle
        .pipe(replace_w_sensible_values)
        .pipe(replace_na)
        .pipe(log_transform_df)
        .pipe(add_AgeDecade)
        # .pipe(utils.add_DebtRatioBin)
        .pipe(add_features_per_dependent)
        .pipe(add_features_distance_from_mean)
        .pipe(add_features_distance_from_median)
        .pipe(add_features_distance_from_std)
    )

    y_pred=model.predict_proba(X_kaggle)[:,1]
    submission.Probability=y_pred

    #Save as csv
    filepath=f'./output/{prefix}_submission.csv'
    submission.to_csv(filepath)
    print(f'Saved to {filepath}')