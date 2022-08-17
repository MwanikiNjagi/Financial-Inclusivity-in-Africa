import config
import preprocessing
import pandas as pd

def main():
    df = pd.read_csv(config.Test)
    #creating bank_account section in index
    df["bank_account"] = ""
    preprocessing.encoding(df)
    df = df.drop(["uniqueid", "year"], axis=1)
    one_hot_encode = ["country","relationship_with_head","marital_status","education_level","job_type"]
    df = pd.get_dummies(df, columns = one_hot_encode)
    print(df.head())
    print(df.dtypes)
    df.to_csv("./input/test_clean.csv", index=False)
    return df

if __name__ == "__main__":
    main()