import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.impute import SimpleImputer
import datetime
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
#########################################################################

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Train Data Info:")
    train_df.info()
    print("\nTrain Data Head:")
    print(train_df.head())
    print("\nTest Data Info:")
    test_df.info()
    print("\nTest Data Head:")
    print(test_df.head())
    print("\nTrain Data NaN sum:")
    print(train_df.isna().sum())
    print("\nTest Data NaN sum:")
    print(test_df.isna().sum())
    return train_df, test_df


def feature_engineering(df, is_train=True, encoders=None, target_encoder=None):
    df = df.copy()
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Policy_Start_Year'] = df['Policy Start Date'].dt.year
    df['Policy_Start_Month'] = df['Policy Start Date'].dt.month
    df['Policy_Start_Day'] = df['Policy Start Date'].dt.day
    df['Policy_Start_Hour'] = df['Policy Start Date'].dt.hour
    df['Policy_Start_Dayofweek'] = df['Policy Start Date'].dt.dayofweek
    base_date = datetime.datetime(2023, 1, 1)
    df['Time_From_Base_Date'] = (df['Policy Start Date'] - base_date).dt.days
    df.drop(columns=['Policy Start Date'], inplace=True)
    df['Age_Income'] = df['Age'] * df['Annual Income']
    df['Dependents_Claims'] = df['Number of Dependents'] * df['Previous Claims']
    df['Insurance_Duration_Age'] = df['Insurance Duration'] * df['Vehicle Age']
    df['Age_Health_Score'] = df['Age'] * df['Health Score']
    df['Income_Credit_Score'] = df['Annual Income'] * df['Credit Score']
    df['Claims_Duration'] = df['Previous Claims'] * df['Insurance Duration']
    df['Age_Dependents'] = df['Age'] * df['Number of Dependents']
    df['Income_Dependents'] = df['Annual Income'] * df['Number of Dependents']
    df['Health_Credit'] = df['Health Score'] * df['Credit Score']
    df['Gender_Marital'] = df['Gender'].astype(str) + '_' + df['Marital Status'].astype(str)
    df['Edu_Occ'] = df['Education Level'].astype(str) + '_' + df['Occupation'].astype(str)
    df['Policy_Location'] = df['Policy Type'].astype(str) + '_' + df['Location'].astype(str)
    df['Smoking_Exercise'] = df['Smoking Status'].astype(str) + '_' + df['Exercise Frequency'].astype(str)
    df['Prop_Feedback'] = df['Property Type'].astype(str) + '_' + df['Customer Feedback'].astype(str)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in numeric_cols:
            if col != 'Premium Amount':
                imputer = SimpleImputer(strategy='median') 
                df[[col]] = imputer.fit_transform(df[[col]])
    if is_train:
        encoders = {} 
        for col in categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
        target_encoder = {}
        for col in categorical_cols:
            target_enc = TargetEncoder(target_type='continuous', cv=5, random_state=42)
            df[col + '_target'] = target_enc.fit_transform(df[[col]], df['Premium Amount'])
            target_encoder[col] = target_enc
    else:
        for col in categorical_cols:
            encoder = encoders[col]
            df[col] = df[col].map(lambda s: encoder.transform([s])[0] if s in encoder.classes_ else 0)
        for col in categorical_cols:
            target_enc = target_encoder[col]
            df[col + '_target'] = target_enc.transform(df[[col]])
    return df, encoders, target_encoder

def train(train_df, target_col, params, n_splits=10):
    X = train_df.drop(columns=[target_col, 'id'])
    y = np.log1p(train_df[target_col])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    def train_fold(train_index, val_index, params):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model = xgb.XGBRegressor(**params, objective='reg:squarederror')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        oof_predictions = model.predict(X_val)
        return model, val_index, oof_predictions
    results = Parallel(n_jobs=-1)(
        delayed(train_fold)(fold, train_index, val_index, params)
        for fold, (train_index, val_index) in enumerate(kf.split(X, y))
    )
    models = []
    oof_predictions = np.zeros(len(train_df))
    for model, val_index, fold_oof_predictions in results:
       models.append(model)
       oof_predictions[val_index] = fold_oof_predictions
    oof_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_predictions)))
    print(f'OOF RMSLE: {oof_rmsle:.4f}')
    return models

def predict(test_df, models):
    X_test = test_df.drop(columns=['id'])
    test_predictions = np.zeros(len(test_df))
    for model in models:
        test_predictions += model.predict(X_test)
    test_predictions /= len(models)  
    return np.expm1(test_predictions) 

def generate_submission_file(test_df, predictions, submission_path):
    submission_df = pd.DataFrame({'id': test_df['id'].astype(np.int32), 'Premium Amount': predictions})
    submission_df.to_csv(submission_path, index=False)
    print(f"submission file created at: {submission_path}")

def run(n_estimators, learning_rate, max_depth, train_df, test_df):
    submission_path = 'handpicked-titan-'+str(n_estimators)+ "-md" + str(max_depth) + '.csv'
    target_col = 'Premium Amount'
    xgb_params = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'n_jobs': -1,
    }
    models = train(train_df, target_col, xgb_params)
    test_predictions = predict(test_df, models)
    generate_submission_file(test_df, test_predictions, submission_path)

def main():    
    train_path = 'train.csv'
    test_path = 'test.csv'
    train_df, test_df = load_data(train_path, test_path)
    print("Data Loaded")

    train_df, encoders, target_encoder = feature_engineering(train_df, is_train=True)
    test_df, _, _ = feature_engineering(test_df, is_train=False, encoders=encoders, target_encoder = target_encoder)
    print("Feature Engineering Done")

    run(2000, 0.01, 8, train_df, test_df)
    # for i in range(1, 20):
    #     print("STARTING RUN   ", i)
    #     run(3000 + i*250, 0.0005+i*0.0001, 8 + i//4, train_df, test_df)
    #     print("RUN COMPLETE: ", i)

if __name__ == "__main__":
    main()