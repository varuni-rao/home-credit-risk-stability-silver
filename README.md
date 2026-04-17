# Home Credit – Credit Risk Model Stability

**Kaggle Silver Medal — Rank 147 / 3,856 (Top 4%)**

Code and experiments for the [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) competition.

## Competition

Predict loan default risk with a focus on **model stability over time**. The evaluation uses a Gini stability metric that penalizes performance degradation across future weeks and rewards consistent predictions — a realistic proxy for production drift in credit scoring.

## Result

- **Private LB Rank:** 147th of 3,856 teams
- **Medal:** Silver
- **Approach:** ensemble of time-aware models (LightGBM, LightAutoML, H2O AutoML) built on a shared utility pipeline

## Repository Structure

```
home-credit-risk-stability-silver/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── utils.py                     # shared functions (from homecreditutility-v5.ipynb)
│   ├── data_prep.py                 # data aggregation pipeline
│   ├── train_lgbm.py                # LightGBM unbalanced ensemble
│   ├── inference_lgbm.py
│   ├── train_lightautoml.py         # LightAutoML CV5
│   ├── inference_lightautoml.py
│   ├── train_h2o.py                 # H2O AutoML
│   └── inference_h2o.py
└── notebooks/
    ├── 00_utils.ipynb               # homecreditutility-v5.ipynb (reference)
    ├── 01_data_prep.ipynb
    ├── 02_lgbm_unbalanced_ensemble.ipynb
    ├── 03_lgbm_inference.ipynb
    ├── 04_lightautoml_cv5.ipynb
    ├── 05_lightautoml_single.ipynb
    ├── 06_lightautoml_inference.ipynb
    ├── 07_h2o_automl_train.ipynb
    └── 08_h2o_automl_submit.ipynb
```

## Key Components

### `src/utils.py`
Central utility module converted from `homecreditutility-v5.ipynb`. It provides:
- Memory-efficient parquet loading with Polars/pandas
- Aggregation helpers for auxiliary tables (bureau, previous, installments)
- Time-based validation split by `WEEK_NUM`
- Feature downcasting and categorical handling
- Common evaluation function for stability metric

All training scripts import from `utils.py` to ensure consistent preprocessing — critical for stability.

### Approaches

1. **LightGBM Unbalanced Ensemble** (`train_lgbm.py`)
   - Uses utils for time-aware splits
   - Class imbalance handling via custom sampling
   - Ensemble of folds

2. **LightAutoML** (`train_lightautoml.py`)
   - Automated feature selection built on utils-processed data
   - 5-fold CV aligned to weeks

3. **H2O AutoML** (`train_h2o.py`)
   - Diversity model trained on same features from utils

## Installation

```bash
git clone https://github.com/<your-username>/home-credit-risk-stability-silver.git
cd home-credit-risk-stability-silver
pip install -r requirements.txt
```

## Reproducibility

1. Download competition data to `data/` (not tracked)
2. Run `python src/data_prep.py` or notebook 01
3. Train: `python src/train_lgbm.py`
4. Inference scripts generate submissions

Notebooks are stored with outputs cleared. For rendered versions, see the Kaggle discussion or run locally.

## License

MIT — see LICENSE file
