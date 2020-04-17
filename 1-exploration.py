#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Plot styling
sns.set_style("whitegrid")
sns.set(font_scale=1.5)

# %% Reading data
train=pd.read_csv('./data/cs-training.csv',index_col=0)
test=pd.read_csv('./data/cs-test.csv',index_col=0)

# %% Checking quality of data
train.info()
# Data types match data definition
# MonthlyIncome and NumberOfDependents have null values

# %% Checking for missing values
train.isna().sum()
print(f'Proportion of null MonthlyIncome= {29731/150000}')
print(f'Proportion of null NumberOfDependents= {3924/150000}')
# 20% missing MonthlyIncome slightly concerning
# 3% missing NumberOfDependents less so
# leave missing for now, potentially fill in later


#%% Checking if the rows with null data contain high predictive power
train['MonthlyIncome_null']=pd.isnull(train.MonthlyIncome)
train['NumberOfDependents_null']=pd.isnull(train.NumberOfDependents)
print(train.groupby('MonthlyIncome_null')['SeriousDlqin2yrs'].mean())
print(train.groupby('NumberOfDependents_null')['SeriousDlqin2yrs'].mean())

# MonthlyIncome and NumberOfDependents have lower mean SeriousDlqin2yrs
# for their null vs non-null groups
# Suggesting predictive power to identify class 1 is low
# We can take a simple approach to fill in missing values (mean/median/mode)

# %% Checking for target class imbalance
ax = sns.countplot(x = train.SeriousDlqin2yrs)

target=train.SeriousDlqin2yrs.value_counts()
for i in target.index:
    print(f'Class {i}, n={target.at[i]}, proportion={target.at[i]/target.sum()}')

# Class 0 aka No SeriousDlqin2 years 
# makes up 93% of the dataset
# severe imbalance
# We'll have to stratify our sample to account for this imbalance

# %% Feature Engineering

#%% What's the baseline correlation amongst features prior to engineering?
def plot_corr_heatmap(df):
    corr=df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))   
    plt.subplots(figsize=(20,15))
    sns.set(font_scale=5)
    chart=sns.heatmap(train.corr(),cmap="PiYG",mask=mask)
    chart.set_xticklabels([])
    return chart

plot_corr_heatmap(train)

# As a baseline, most of features show weak correlation with our target

# %% Plotting feature distributions
def check_distribution_boxplot(feature):
    fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))
    sns.set(font_scale=1)
    
    plot00=sns.distplot(feature,ax=axes[0],color='m')
    plt.tight_layout()

    plot01=sns.boxplot(train[feature,ax=axes[1],orient = 'v',color='c')
    plt.tight_layout()


# %%
for feature_name in train.columns:
    check_distribution_boxplot(train[feature_name])

# %%
