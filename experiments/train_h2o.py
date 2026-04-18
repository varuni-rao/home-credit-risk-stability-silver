# Experimental H2O ensemble — explored for diversity
# Final silver submission used LightAutoML (see train_lightautoml.py)

# Importing libraries and loading data for H2O model
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import h2o
from h2o.automl import H2OAutoML
import pickle
import joblib
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import gc
import warnings
warnings.filterwarnings("ignore")
from src import utils as hcu

def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = base.loc[:, ["WEEK_NUM", "target", "predict"]].sort_values("WEEK_NUM").groupby("WEEK_NUM")[["target", "predict"]].apply(lambda x: 2*roc_auc_score(x["target"], x["predict"])-1).tolist()
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std

if __name__ == "__main__":
    h2o.init()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.parquet")
    parser.add_argument("--out_dir", default="models/h2o")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    train_df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    train_hf = h2o.H2OFrame(train_df)
    
    y_feature = 'target'
    x_feature = list(train_hf.columns)
    x_feature.remove('target'); x_feature.remove('case_id'); x_feature.remove('WEEK_NUM')
    
    train_hf[y_feature] = train_hf[y_feature].asfactor()
    
    aml = H2OAutoML(max_models=11, max_runtime_secs=14400, max_runtime_secs_per_model=1200, seed=13, include_algos=["GBM","StackedEnsemble"], verbosity='info', stopping_metric='auc')
    aml.train(x=x_feature, y=y_feature, training_frame=train_hf)
    
    for i in range(min(3, aml.leaderboard.nrows)):
        mod_id = aml.leaderboard[i,0]
        model = h2o.get_model(mod_id)
        h2o.save_model(model=model, path=args.out_dir, force=True)
        if i==0: best_model_id=mod_id
    
    with open('data/processed/h2o_features.pkl','wb') as f: 
        pickle.dump(x_feature,f)
    
    y_pred = aml.leader.predict(train_hf[x_feature])
    
    y_true = train_hf['target'].as_data_frame().values.ravel()
    
    y_pred_vals = y_pred['p1'].as_data_frame().values.ravel()
    
    print(f'The AUC score on the train set is: {roc_auc_score(y_true, y_pred_vals)}')
    
    pred_df = train_hf[["WEEK_NUM","target"]].as_data_frame()
    pred_df["predict"]=y_pred_vals
    
    stability_score_train = gini_stability(pred_df)
    print(f'The stability score on the train set is: {stability_score_train}')
    print(f'Best model saved to {args.out_dir}/{best_model_id}')
