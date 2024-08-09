import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline

from l0_recorder import ExperimentRecorder
from l0_settings import Settings

global_settings = Settings()

# Load data

stacking_models_list = global_settings.get_stacking_models_list()
print(f"stacking_models_list: {stacking_models_list}")

print(f"Loading oofs...")
oofs_train = []
oofs_inference = []
for stacking_model in stacking_models_list:
    if stacking_model in global_settings.black_listed_models:
        print(f"Skipping {stacking_model} because it is blacklisted.")
        continue

    oofs_train.append(pd.read_csv(global_settings.oof_dir / f'{stacking_model}_train.csv.gz',
                                  index_col=global_settings.id_col))
    oofs_inference.append(pd.read_csv(global_settings.oof_dir / f'{stacking_model}_inference.csv.gz',
                                      index_col=global_settings.id_col))

oofs_train_df = pd.concat(oofs_train, axis=1)
y = pd.read_csv(global_settings.raw_features_dir /
                f'train_features_with_scores_v{global_settings.get_featureset_version()}'
                f'.csv.gz')[global_settings.target_col]
y_as_str = y.astype(str)
oofs_train_df[global_settings.target_col] = y
oofs_inference_df = pd.concat(oofs_inference, axis=1)

print(f"Loading train and test data for meta model training and inference...")
print(f"oofs_train_df.shape: {oofs_train_df.shape}")
print(f"columns in oofs_train_df: {oofs_train_df.columns}")

print(f"oofs_inference_df.shape: {oofs_inference_df.shape}")
print(f"columns in oofs_inference_df: {oofs_inference_df.columns}")

oof_columns = [col for col in oofs_train_df.columns if col not in [global_settings.id_col, global_settings.target_col] +
               global_settings.get_black_listed_meta_features()]

# Prepare data for training
if global_settings.use_stratified_folds or True:
    skf = StratifiedKFold(n_splits=global_settings.l3_n_folds, shuffle=True,
                          random_state=global_settings.global_seed + global_settings.random_seed_offsets[2])
else:
    skf = KFold(n_splits=global_settings.l3_n_folds, shuffle=True,
                random_state=global_settings.global_seed + global_settings.random_seed_offsets[2])

# Train model and perform inference
inferences = []
validation_rmses = []

y_val = np.zeros(len(oofs_train_df))
y_hat_val = np.zeros(len(oofs_train_df))

for fold, (train_idx, valid_idx) in enumerate(skf.split(oofs_train_df, y_as_str)):
    X_train, X_valid = oofs_train_df.iloc[train_idx][oof_columns], oofs_train_df.iloc[valid_idx][oof_columns]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    X_test = oofs_inference_df[oof_columns]

    preprocessing = make_pipeline(
        # FunctionTransformer(np.expm1, validate=False),
        # PolynomialFeatures(degree=2, include_bias=False),
        # SplineTransformer(),
        # SelectKBest(k=1000, score_func=f_regression),
        SelectKBest(k='all', score_func=f_regression),
    )

    X_train = preprocessing.fit_transform(X_train, y_train)
    X_valid = preprocessing.transform(X_valid)
    X_test = preprocessing.transform(X_test)

    model = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=10000, tol=1e-6)

    model.fit(X_train, y_train)

    y_hat = model.predict(X_valid)

    # y_hat = np.clip(y_hat, 1.5, 5.5)

    y_val[valid_idx] = y_valid
    y_hat_val[valid_idx] = y_hat

    val_rmse = mean_squared_error(y.iloc[valid_idx], y_hat, squared=False)
    validation_rmses.append(val_rmse)
    print(f"Fold {fold + 1} RMSE: {np.round(val_rmse, 4)}")
    inferences.append(model.predict(X_test))

# Save inferences for submission to Kaggle competition (average of folds); include the id column
inferences_df = pd.DataFrame(np.mean(inferences, axis=0), columns=[global_settings.target_col],
                             index=oofs_inference_df.index)

# Make the submission file
inferences_df.to_csv(global_settings.get_meta_model_oofs_dir() / 'submission_huber.csv.gz', index=True)

script_name = os.path.basename(__file__).split('.')[0]

# Correlation between oofs and target
corr = np.corrcoef(y_hat_val, y_val)[0, 1]

recorder = ExperimentRecorder()
recorder.add_record(model_name=script_name,
                    rmse_mean=np.mean(validation_rmses),
                    rmse_std=np.std(validation_rmses),
                    correlation_with_target=corr,
                    featureset_version=-1,
                    n_folds=global_settings.l3_n_folds,
                    random_seed=global_settings.global_seed + global_settings.random_seed_offsets[2], )

print("Done!")

print(f"\nNumber of oofs: {len(oofs_train)}")
print(f"Number of blacklisted models/oofs: {len(global_settings.black_listed_models)}")

print(f"\nMeta model training and inference done!\n"
      f"Validation RMSE: {np.round(np.mean(validation_rmses), 4)} "
      f"+- {np.round(np.std(validation_rmses), 4)} on {global_settings.l2_n_folds} folds")

# Save the validation predictions
meta_model_df = pd.DataFrame({global_settings.id_col: oofs_train_df.index, 'y_hat': y_hat_val})
meta_model_df.to_csv(global_settings.get_meta_model_oofs_dir() / f'val_huber.csv.gz', index=False)

print("\nDone!")
