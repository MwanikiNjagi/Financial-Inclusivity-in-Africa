import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import config

def main():
    df = pd.read_csv(config.Train)
    df = df.drop(["uniqueid","year"], axis=1)
    encoding(df)
    print(df.head())
    print(df.describe())
    print(df.dtypes)
    one_hot_encode = ["country","relationship_with_head","marital_status","education_level","job_type"]
    df = pd.get_dummies(df, columns = one_hot_encode)
    scaler = MinMaxScaler(feature_range=(0,1))
    df = scaler.fit_transform(df)
    #print(df.isnull().sum()) There are no null values
    pd.DataFrame(df).to_csv("./input/train_clean.csv", index=False)
    return df

def encoding(df):
    #Label encoding for booleans
    LE = LabelEncoder()
    le_encode = ["bank_account", "location_type", "cellphone_access", "gender_of_respondent"]
    df[le_encode] = df[le_encode].apply(LE.fit_transform)
    return df


#It will be necessary to attempt model development without encoding with Catboost and LGBM to see the results




if __name__ == "__main__":
    main()
