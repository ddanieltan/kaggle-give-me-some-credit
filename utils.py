import numpy as np
import pandas as pd

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
    df['MonthlyIncome']=df['MonthlyIncome'].fillna(1.0)
    df['NumberOfDependents']=df['NumberOfDependents'].fillna(1)

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

# def add_DebtRatioBin(df_):
#     '''
#     Input: dataframe
#     Output: dataframe with feature binning DebtRatio
#     '''
#     df=df_.copy()
#     df['DebtRatio']=pd.cut(
#         x=df['DebtRatio'],
#         bins=[-0.0001,0.33,0.43,1.0],
#         labels=[0,0.33,1]
#     ) 
#     return df

def add_features_per_dependent(df_):
    '''
    Input: dataframe
    Output: dataframe with additional 4 transformed features
    '''
    df=df_.copy()
    df['MonthlyIncomePerDependent']=df['MonthlyIncome']/df['NumberOfDependents']
    df['NumberRealEstateLoansOrLinesPerDependent']=df['NumberRealEstateLoansOrLines']/df['NumberOfDependents']
    df['NumberOfOpenCreditLinesAndLoansPerDependent']=df['NumberOfOpenCreditLinesAndLoans']/df['NumberOfDependents']
    df['NumberOfTime30-59DaysPastDueNotWorsePerDependent']=df['NumberOfTime30-59DaysPastDueNotWorse']/df['NumberOfDependents']

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
    
# Plotting functions

def plot_corr_heatmap(df):
    corr=df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))   
    plt.subplots(figsize=(20,15))
    sns.set(font_scale=5)
    chart=sns.heatmap(train.corr(),cmap="PiYG",mask=mask)
    chart.set_xticklabels([])
    return chart


# Checking functions

def check_proportion_of_classes(ser):
    '''
    Input: Series
    Output: None

    Prints proportion of class 1
    '''
    vc=df.value_counts()
    print(vc)
    print(f'Proportion of class 1: {vc[1]/vc.sum()}')

# Submission function

def create_submission(model):
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
    prefix=str(np.random.randint(1000,9999))
    filepath=f'./output/{prefix}_submission.csv'
    submission.to_csv(filepath)
    print(f'Saved to {filepath}')