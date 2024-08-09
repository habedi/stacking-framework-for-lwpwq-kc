import os

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

from l0_recorder import ExperimentRecorder
from l0_settings import Settings

global_settings = Settings()

# Load data

train_x_y = pd.read_csv(global_settings.raw_features_dir /
                        f'train_features_with_scores_v{2}.csv.gz')
test_x = pd.read_csv(global_settings.raw_features_dir /
                     f'test_features_v{2}.csv.gz')

# Have a look at the data we have
print(f"train_x_y.shape: {train_x_y.shape}")
print(f"test_x.shape: {test_x.shape}")

# Prepare data for training
if global_settings.use_stratified_folds:
    skf = StratifiedKFold(n_splits=global_settings.l2_n_folds, shuffle=True, random_state=global_settings.global_seed)
else:
    skf = KFold(n_splits=global_settings.l2_n_folds, shuffle=True, random_state=global_settings.global_seed)

train_x = train_x_y.drop([global_settings.id_col, global_settings.target_col], axis=1)
train_y = train_x_y[global_settings.target_col]
train_y_as_str = train_y.astype(str)

oof_train = np.zeros(len(train_x))
oof_test = np.zeros(len(test_x))

validation_rmses = []
# Train model and predict
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y_as_str)):
    model = cb.CatBoostRegressor(iterations=1500,
                                 learning_rate=0.025,
                                 depth=5,
                                 eval_metric='RMSE',
                                 random_seed=global_settings.global_seed,
                                 bagging_temperature=0.5,
                                 od_type='Iter',
                                 metric_period=50,
                                 od_wait=15,
                                 boosting_type='Plain',
                                 subsample=0.8,
                                 # reg_lambda=0.02,
                                 # grow_policy='Depthwise',
                                 verbose=100,
                                 border_count=1024,
                                 # random_strength=0.0,
                                 # l2_leaf_reg=3,
                                 # one_hot_max_size=128,
                                 )

    model.fit(train_x.iloc[train_idx], train_y.iloc[train_idx],
              eval_set=(train_x.iloc[valid_idx], train_y.iloc[valid_idx]),
              use_best_model=True,
              early_stopping_rounds=200
              )

    y_hat = model.predict(train_x.iloc[valid_idx])
    oof_train[valid_idx] = y_hat
    val_rmse = mean_squared_error(train_y.iloc[valid_idx], y_hat, squared=False)
    validation_rmses.append(val_rmse)
    print(f"Fold {fold + 1} RMSE: {np.round(val_rmse, 4)}")

print(f"\nTraining finished for {fold + 1} folds with "
      f"mean validation RMSE of {np.round(np.mean(validation_rmses), 4)} "
      f"with std {np.round(np.std(validation_rmses), 4)}")

oof_test = model.predict(test_x.drop([global_settings.id_col], axis=1))

# Get script name to use as column name and file name
script_name = os.path.basename(__file__).split('.')[0]

# Save the results as dataframes
oof_train_df = pd.DataFrame(
    {global_settings.id_col: train_x_y[global_settings.id_col], f'oof_{script_name}': oof_train})
oof_test_df = pd.DataFrame({global_settings.id_col: test_x[global_settings.id_col], f'oof_{script_name}': oof_test})

# Save the results as csv files (create the path if it doesn't exist)
global_settings.oof_dir.mkdir(parents=True, exist_ok=True)

# add module name to the file name and column name
oof_train_df.to_csv(global_settings.oof_dir / f'{script_name}_train.csv.gz', index=False)
oof_test_df.to_csv(global_settings.oof_dir / f'{script_name}_inference.csv.gz', index=False)

# Correlation between oofs and target
corr = np.corrcoef(oof_train, train_y)[0, 1]

recorder = ExperimentRecorder()
recorder.add_record(model_name=script_name,
                    rmse_mean=np.mean(validation_rmses),
                    rmse_std=np.std(validation_rmses),
                    correlation_with_target=corr,
                    featureset_version=2)

print("Done!")
