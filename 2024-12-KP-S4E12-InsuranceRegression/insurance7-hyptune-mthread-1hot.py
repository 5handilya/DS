import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self.params = params
        self.model = xgb.XGBRegressor(enable_categorical=True)  # Enable categorical features
        self.model.set_params(**params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def set_params(self, **params):
        self.params.update(params)
        self.model.set_params(**params)
        return self
    
    def get_params(self, deep=True):
        return self.params

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
    
    df['Annual Income'] = np.log1p(df['Annual Income'])
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
    # One-Hot Encoding for categorical variables
    categorical_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']
    df[categorical_cols] = df[categorical_cols].astype('category')
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    
    return df

def train(train_df, target_col, params, n_splits=10):
    X = train_df.drop(columns=[target_col, 'id'])
    y = np.log1p(train_df[target_col])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def train_fold(train_index, val_index, params):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model = xgb.XGBRegressor(**params, objective='reg:squarederror', enable_categorical=True)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        oof_predictions = model.predict(X_val)
        return model, val_index, oof_predictions
    
    results = Parallel(n_jobs=8)(
        delayed(train_fold)(train_index, val_index, params)
        for train_index, val_index in kf.split(X, y)
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

def run(params, train_df, test_df):
    submission_path = '8-hyp-1hot.csv'
    target_col = 'Premium Amount'
    
    train_df = feature_engineering(train_df, is_train=True)
    test_df = feature_engineering(test_df, is_train=False)
    
    models = train(train_df, target_col, params)
    predictions = predict(test_df, models)
    generate_submission_file(test_df, predictions, submission_path)

def hyperparameter_tuning(train_df, target_col):
    X = train_df.drop(columns=[target_col, 'id'])
    y = np.log1p(train_df[target_col])
    
    xgb_model = XGBRegressorWrapper(objective='reg:squarederror', n_jobs=-1, random_state=42)
    
    param_dist = {
        'n_estimators': [2750, 3750, 4750, 5750, 6250, 7250, 8000, 8250, 8750],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [4,6,7,8,9],
        'min_child_weight': [1,3,5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=50, scoring='neg_mean_squared_log_error', cv=kf, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best RMSLE score: {np.sqrt(-random_search.best_score_)}")
    
    return random_search.best_params_

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    best_params = hyperparameter_tuning(train_df, 'Premium Amount')
    run(best_params, train_df, test_df)

if __name__ == "__main__":
    main()
