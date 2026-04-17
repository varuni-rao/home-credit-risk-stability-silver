# Home Credit – Credit Risk Model Stability

**Kaggle Silver Medal — Rank 147 / 3,856 (Top 4%)**

Code and experiments for the [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) competition.

## Competition

Predict loan default risk with a focus on **model stability over time**. The evaluation uses a Gini stability metric that penalizes performance degradation across future weeks and rewards consistent predictions — a realistic proxy for production drift in credit scoring.

## Result

- **Private LB Rank:** 147th of 3,856 teams
- **Medal:** Silver
- **Approach:** ensemble of time-aware models (LightGBM, LightAutoML, H2O AutoML) trained with unbalanced sampling and week-based validation

## Repository Structure

```
home-credit-risk-stability-silver/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_prep.ipynb               
│   ├── 02_lightgbm_unbalanced.ipynb      
│   ├── 03_lightgbm_inference.ipynb       
│   ├── 04_lightautoml_cv5.ipynb           
│   ├── 05_lightautoml_single.ipynb        
│   ├── 06_lightautoml_inference.ipynb     
│   ├── 07_h2o_automl_train.ipynb          
│   └── 08_h2o_automl_submit.ipynb         
└── src/                                   # (recommended) converted .py versions for reproducibility
    ├── data_prep.py
    ├── train_lgbm.py
    ├── train_lightautoml.py
    └── train_h2o.py
```

Rename your original notebooks as shown for clarity. The `src/` folder is optional but recommended for clean imports and Kaggle re-runs.

## Approaches Explored

1. **LightGBM Unbalanced Ensemble**
   - File: `02_lightgbm_unbalanced.ipynb`
   - Handles class imbalance via custom sampling
   - Time-based validation using WEEK_NUM to mirror stability metric
   - Ensemble of folds for reduced variance

2. **LightAutoML (CV5)**
   - Files: `04_lightautoml_cv5.ipynb`, `05_lightautoml_single.ipynb`
   - Automated feature selection and stacking
   - 5-fold time-aware CV; inference notebook generates out-of-fold stability scores

3. **H2O AutoML**
   - Files: `07_h2o_automl_train.ipynb`, `08_h2o_automl_submit.ipynb`
   - Baseline automl with leaderboard models (GBM, XGBoost, DeepLearning)
   - Used for diversity in final ensemble

4. **Data Preparation**
   - File: `01_data_prep.ipynb`
   - Aggregates auxiliary tables (bureau, previous applications, installments) to case_id level
   - Memory optimization with categorical downcasting

## Installation

```bash
git clone https://github.com/<your-username>/home-credit-risk-stability-silver.git
cd home-credit-risk-stability-silver
pip install -r requirements.txt
```

Core dependencies: `lightgbm>=4.0`, `lightautoml`, `h2o`, `pandas`, `polars`, `scikit-learn`, `numpy`

## How to Reproduce

1. Download data from Kaggle competition page into `data/` (not tracked)
2. Run data prep notebook first
3. Train models in order: LightGBM → LightAutoML → H2O
4. Use inference notebooks to generate test predictions
5. Ensemble predictions (simple weighted average worked best for stability)

## Key Learnings

- Week-based splits predicted LB stability far better than random KFold
- Unbalanced sampling improved recall on late weeks without hurting early performance
- Simple aggregations (mean, max, last) were more stable than high-cardinality interactions
- Ensembling diverse frameworks (tree + automl) reduced week-to-week variance

## Recommendations for Cleanup

- Clear notebook outputs before committing (`Kernel → Restart & Clear Output`)
- Convert stable notebooks to `.py` with `jupyter nbconvert --to script`
- Add `.gitignore` for `data/`, `models/`, `.ipynb_checkpoints/`
- Pin versions in `requirements.txt`

## License

MIT

## Acknowledgements

Home Credit and Kaggle for the dataset and stability-focused evaluation.
