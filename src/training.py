from lib2to3.pgen2.pgen import DFAState
import optuna
import config
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def main():
    df = pd.read_csv(config.Train)
    print(df.head())
    print(df.dtypes)
    return df
if __name__ == "__main__":
    main()