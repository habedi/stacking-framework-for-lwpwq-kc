import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from l0_recorder import ExperimentRecorder
from l0_settings import Settings

global_settings = Settings()

meta_models_weights = {
    "lgbm": 0.2,
    "xgb": 0.2,
    "huber": 0.2,
    "cb": 0.2,
    "tpot": 0.2,
}

meta_models_weights_and_predictions = dict()

for model_name, weight in meta_models_weights.items():

    if weight > 0:
        print(f"Loading {model_name}...")

        meta_models_weights_and_predictions[model_name] = (weight,
                                                           pd.read_csv(global_settings.get_meta_model_oofs_dir() /
                                                                       f'submission_{model_name}.csv.gz',
                                                                       index_col=global_settings.id_col),
                                                           pd.read_csv(global_settings.get_meta_model_oofs_dir() /
                                                                       f'val_{model_name}.csv.gz',
                                                                       index_col=global_settings.id_col),
                                                           )
    else:
        print(f"Skipping {model_name} because weight is {weight}.")

final_predictions = np.zeros(len(meta_models_weights_and_predictions["lgbm"][1]))
val_preds = np.zeros(len(meta_models_weights_and_predictions["lgbm"][2]))
for model_name, w_p in meta_models_weights_and_predictions.items():
    weight, inference, val = w_p[0], w_p[1], w_p[2]
    print(f"model_name: {model_name}")
    print(f"weight: {weight}")

    if weight > 0:
        print(f"Using {model_name} with weight {weight} for final predictions.")
        final_predictions += weight * inference[global_settings.target_col].values
        val_preds += weight * val['y_hat'].values
    else:
        print(f"Skipping {model_name} because weight is {weight}.")

    print(50 * "-")

y = pd.read_csv(
    global_settings.raw_features_dir / f'train_features_with_scores_v{global_settings.get_featureset_version()}.csv.gz')[
    global_settings.target_col]

# Print the final RMSE and correlation with target
rmse = mean_squared_error(y, val_preds, squared=False)
print(f"Final RMSE: {np.round(rmse, 4)}")
corr = np.corrcoef(val_preds, y)[0, 1]
print(f"Final correlation with target: {np.round(corr, 4)}")

# Make the submission file
inferences_df = pd.DataFrame(final_predictions, columns=[global_settings.target_col],
                             index=meta_models_weights_and_predictions["lgbm"][1].index)

# Make the submission file
inferences_df.to_csv(global_settings.output_dir / 'submission.csv.gz', index=True)

script_name = os.path.basename(__file__).split('.')[0]

recorder = ExperimentRecorder()
recorder.add_record(model_name=script_name,
                    rmse_mean=rmse,
                    rmse_std=0,
                    correlation_with_target=corr,
                    featureset_version=-2,
                    n_folds=global_settings.l3_n_folds, )

print("Done!")
