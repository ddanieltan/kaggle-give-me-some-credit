# kaggle-give-me-some-credit

## Overview
This is my attempt of the Kaggle classification problem - [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit).

My best AUC score achieved was **0.84876**, which would place me ~**658** on the private leaderboard.

## Files

1-exploration.py : This code explores the dataset and identifies the target class imbalance, potential feature engineering and missing data that needs to be addressed.

2-model.py : This code implements the target and feature engineering, followed by testing 3 different models (Logistic Regression, Random Forests and LightGBM). This code concludes with a first submission to Kaggle using a LightGBM model with mostly default parameters. This submission scored 0.84579, placing 663 on the leaderboard.

3-tuning.py : This code implements parameter tuning using RandomizedSearchCV on 7 folds, 50 candidates of parameters. Additionally, this code also incorporates additional features from studying more about the domain of the problem (credit risk/ratings). The final submission after this tuning scored 0.84876, placing 658 on the leaderboard.

## Note
I do my work directly in python files, utilising the [VSCode feature]((https://code.visualstudio.com/docs/python/jupyter-support)) that allows me render a jupyter motebook codeblock by prefixing my code with '#%%'. This explains why my code is littered with '#%%'.


## Progress of Kaggle AUC score and leaderboard placement

| AUC Score  | Leaderboard Position |
|------------|----------------------|
| 0.84876 | 658|
|0.84722 | 664|
|0.84579 | 672|