# train_lightautoml.py — Silver-medal solution
# Home Credit Credit Risk Model Stability

"""LightAutoML with time-aware validation (group=WEEK_NUM)."""

""""Import dependencies and set up parameters for lightautoml"""
import numpy as np # linear algebra
import pandas as pd # data processing

# library for metrics
from sklearn.metrics import roc_auc_score

# library for AUTOML
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task

# library to read and write files
import pickle

# library to save and load models
import joblib

import torch
import argparse

# display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# library for garbage collection
import gc  # since the data is huge here, regularly cleaning up will free up memory

# library to catch and ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""Function to calculate gini stability score for the predictions"""

def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = base.loc[:, ["WEEK_NUM", "target", "predict"]]\
        .sort_values("WEEK_NUM")\
        .groupby("WEEK_NUM")[["target", "predict"]]\
        .apply(lambda x: 2*roc_auc_score(x["target"], x["predict"])-1).tolist()

    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


"""Since the test dataset is huge, we will predict in batches and then concatenate the results to get the final predictions for the test dataset"""

def predict_proba_in_batches(model, data, batch_size=100000):
    num_samples = len(data)
    num_batches = int(np.ceil(num_samples / batch_size))
    probabilities = np.zeros((num_samples,))

    for batch_idx in range(num_batches):
        print(f"Processing batch: {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        X_batch = data.iloc[start_idx:end_idx].reset_index(drop=True)
        batch_probs = model.predict(X_batch).data[:, 0]
        probabilities[start_idx:end_idx] = batch_probs

    return probabilities

if __name__ == "__main__":
    """Read the data and set up the parameters for lightautoml"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.parquet")
    parser.add_argument("--cat_cols", default="data/processed/train_cat_cols.pkl")
    args = parser.parse_args()

    train_df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)

    # setting categorical data as categories
    with open(args.cat_cols, 'rb') as f:
        cat_cols = pickle.load(f)
            
    train_df[cat_cols] = train_df[cat_cols].astype('category')
    train_df.info()

    """Set up parameters for lightautoml and train the model and predict on train dataset to find gini stability score"""
    
    #Parameters and fix torch no of threads and numpy seeds
    N_FOLDS = 5 # 5-fold cv
    N_THREADS = 4  #threads
    RANDOM_STATE=13 # fixed random state
    TIMEOUT = 10*3600 #Time

    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(N_THREADS)


    #initiate the task
    task = Task('binary', metric='auc')

    #feature selection
    roles = {
        'target':'target',
        'group': "WEEK_NUM",
        'drop':['case_id', 'WEEK_NUM'],
    }

    #Automl
    automl = TabularUtilizedAutoML(task = task,
                                timeout=TIMEOUT,
                                cpu_limit=N_THREADS,
                                gpu_ids = 'all',
                                reader_params = {'n_jobs':N_THREADS, 'cv': N_FOLDS},
                                general_params = {'use_algos':[['lgb', 'lgb_tuned','cb', 'cb_tuned']]},
                                tuning_params = {'max_tuning_time':60*60},)
                                
    #prediction
    pred = automl.fit_predict(train_df,roles=roles, verbose=2)
    print('pred:\n{}\nShape = {}'.format(pred[:10],pred.shape))

    # save automl model
    joblib.dump(automl, 'automl.joblib')

    """Predict on the train dataset to find gini stability score. This is just to check the stability of the model and not the final predictions for the test dataset"""

    # predict using the trained gbm model
    train_cols = train_df.columns.to_list()
    train_cols.remove('case_id')
    train_cols.remove('target')
    train_cols.remove('WEEK_NUM')
    y_pred = pd.Series(predict_proba_in_batches(automl, train_df[train_cols]), index=train_df.index)

    # display AUC score for the train and validation datasets
    print(f'The AUC score on the train set is: {roc_auc_score(train_df["target"], y_pred)}')

    pred_df = train_df[["WEEK_NUM", "target"]].copy()
    pred_df["predict"] = y_pred

    # finding the stability scores for train and validation datasets
    stability_score_train = gini_stability(pred_df)

    # display the stability scores for train and validation dataset
    print(f'The stability score on the train set is: {stability_score_train}')