# kaggle-give-me-some-credit

## Overview
This is my attempt of the Kaggle classification problem - [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit).

My best AUC score achieved was **0.84876**, which would place me ~**658** on the private leaderboard.

## Usage
```
conda create --name credit --file requirements.txt
python main.py
```

## File structure
- /data : Input data provided by Kaggle
- /dev : Development files used to explore problem and evaluate models
- /output : Final submission (csv), feature list, AUC and PRC plots are saved here
- utils.py : Collection of utility functions
- requirements.txt : Project dependencies
- main.py : Main script

## Progress of Kaggle AUC score and leaderboard placement

| AUC Score  | Leaderboard Position |
|------------|----------------------|
|0.86132| 511 |
|0.86112| 511|
|0.84876 | 658|
|0.84722 | 664|
|0.84579 | 672|