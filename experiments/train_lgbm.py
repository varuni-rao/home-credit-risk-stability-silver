# Experimental LightGBM ensemble — explored for diversity
# Final silver submission used LightAutoML (see train_lightautoml.py)

""" Importing libraries and loading data for LightGBM model with unbalanced ensemble approach. """
import numpy as np # linear algebra
import pandas as pd # data processing

# import further libraries
import matplotlib.pyplot as plt
import seaborn as sns

# library for cross validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# library for metrics
from sklearn.metrics import roc_auc_score

# library for LightGBM Model
import optuna.integration.lightgbm as lgb
import lightgbm as lgbm
from sklearn.ensemble import VotingClassifier

# library to read and write files
import pickle

# library to save and load models
import joblib

# display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# library for garbage collection
import gc  # since the data is huge here, regularly cleaning up will free up memory

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin

# library to catch and ignore warnings
import warnings
warnings.filterwarnings("ignore")

# library for utility script containing various pipelines
from src import utils as hcu #version7

"""Batch Prediction Function"""

def predict_proba_in_batches(model, data, batch_size=100000):
    num_samples = len(data)
    num_batches = int(np.ceil(num_samples / batch_size))
    probabilities = np.zeros((num_samples,))

    for batch_idx in range(num_batches):
        print(f"Processing batch: {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        X_batch = data.iloc[start_idx:end_idx]
        batch_probs = model.predict_proba(X_batch)[:, 1]
        probabilities[start_idx:end_idx] = batch_probs
        gc.collect()

    return probabilities


"""A custom voting model that takes a list of fitted estimators and implements the predict and predict_proba methods by averaging the predictions from all the estimators. This allows us to create an ensemble model that combines the predictions of multiple LightGBM models trained on different folds of the data."""

class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
    
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
    
"""Function to determine gini stability of the model predictions across time. The function calculates the Gini coefficient for each week, fits a linear regression to the Gini coefficients over time, and calculates the standard deviation of the residuals. The final stability score is a combination of the average Gini coefficient, the slope of the fitted line (penalizing negative slopes), and the standard deviation of the residuals (penalizing higher variability)."""
def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = base.loc[:, ["WEEK_NUM", "target", "predict"]]\
        .sort_values("WEEK_NUM")\
        .groupby("WEEK_NUM")[["target", "predict"]]\
        .apply(lambda x: 2*roc_auc_score(x["target"], x["predict"])-1).tolist()

    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    print("y_hat = {}*x + {}".format(a,b))
    residuals = y - y_hat
    res_std = np.std(residuals)
    print("Residual Std. Dev:",res_std)
    avg_gini = np.mean(gini_in_time)
    print("Mean Gini in time:", avg_gini)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std
        
def main():

    """Create directories to save models and processed data if they don't exist."""
    import os
    os.makedirs('models/lgbm', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    """Load data**"""
    # loading data into train_df
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.parquet")
    parser.add_argument("--cat_cols", default="data/processed/cat_cols.pkl")
    args = parser.parse_args()
    
    train_df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    with open(args.cat_cols, 'rb') as f:
        cat_cols = pickle.load(f)
            
    train_df[cat_cols] = train_df[cat_cols].astype('category')
    train_df.info()

    train_df.sample(5)

    """Data Preparation"""
    y = train_df["target"]
    weeks = train_df["WEEK_NUM"]
    X = train_df.drop(columns=["target", "case_id", "WEEK_NUM"], axis = 1)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

    X.shape, y.shape

    # garbage collection
    hcu.MemoryOptimizer.CleanMemory()
    gc.collect()

    """Build CV LightGBM hypertuned Models"""

    # baseline parameters for comparing changes done to the features (0.554)
    params = {"boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "max_depth": 8,
            "max_bin": 255,
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "colsample_bytree": 0.8,
            "colsample_bynode": 0.8,
            #  "class_weight": "balanced",
            "verbose": -1,
            "random_state": 42,
            "device": "gpu" if lgbm.compat._IS_GPU_ENABLED else "cpu",}


    # changing categories to str type
    X[cat_cols] = X[cat_cols].astype(str)


    fitted_models_lgb = []
    cv_scores_lgb = []
    oof_pred = np.zeros(X.shape[0])

    for idx_train, idx_valid in cv.split(X, y, groups=weeks):#
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]# 
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
        
        X_train[cat_cols] = X_train[cat_cols].astype("category")
        X_valid[cat_cols] = X_valid[cat_cols].astype("category")
        
        model = lgbm.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_valid, y_valid)],
            callbacks = [lgb.log_evaluation(100), lgb.early_stopping(100)] )
        
        fitted_models_lgb.append(model)
        val_pred = model.predict_proba(X_valid)[:, 1]
        oof_pred[idx_valid] = val_pred  
        auc_score = roc_auc_score(y_valid, val_pred)
        cv_scores_lgb.append(auc_score)
        gc.collect()
        
    print("CV AUC scores: ", cv_scores_lgb)
    print("Maximum CV AUC score: ", max(cv_scores_lgb))
    roc_auc_oof = roc_auc_score(y, oof_pred)
    print("CV roc_auc_oof: ", roc_auc_oof)


    # save models
    joblib.dump(fitted_models_lgb, 'models/lgbm/fitted_models_lgb.joblib')
    joblib.dump((train_df.columns, cat_cols), 'models/lgbm/train_cat_columns.pkl')


    # save consistent files
    
      # rename to match inference
    joblib.dump(cat_cols, 'data/processed/lgbm_cat_cols.pkl')
    joblib.dump(X.columns.tolist(), 'data/processed/lgbm_features.pkl')

    oof_models_dict = [(str(i), model) for i, model in enumerate(fitted_models_lgb)]

    vmodel2 = VotingClassifier(
        estimators=oof_models_dict, 
        voting = 'soft'
    )

    vmodel2.estimators_ = fitted_models_lgb

    # save model
    joblib.dump(vmodel2, 'models/lgbm/lgbm_voting.joblib')

    X[cat_cols] = X[cat_cols].astype("category")

    # predict using the trained gbm model
    y_pred = pd.Series(predict_proba_in_batches(vmodel2, X), index=X.index)

    # display AUC score for the train and validation datasets
    print(f'The AUC score on the train set is: {roc_auc_score(y, y_pred)}')

    pred_df = train_df[["WEEK_NUM", "target"]].copy()
    pred_df["predict"] = y_pred

    # finding the stability scores for train and validation datasets
    stability_score_train = gini_stability(pred_df)

    # display the stability scores for train and validation dataset
    print(f'The stability score on the train set is: {stability_score_train}')

    plt.figure(figsize = (15,10))
    sns.kdeplot(x=train_df['target'], color = 'blue', label = 'target')
    sns.kdeplot(x=y_pred, color = 'red', label = 'prediction')
    plt.legend(loc= 'upper right')
    plt.show()

if __name__ == "__main__":
    main()