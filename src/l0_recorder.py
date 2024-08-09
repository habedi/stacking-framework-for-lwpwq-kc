import numpy as np
import pandas as pd

from l0_settings import Settings


class ExperimentRecorder:
    def __init__(self, settings=Settings(), recreate_records_file=False):
        self.settings = settings
        self.experiment_records_file_path = self.settings.output_dir / self.settings.experiment_records_file_name
        self.experiment_records_file_path.parent.mkdir(exist_ok=True, parents=True)

        if recreate_records_file and self.experiment_records_file_path.exists():
            self.experiment_records_file_path.unlink()

        self.experiment_records = pd.read_csv(
            self.experiment_records_file_path) if self.experiment_records_file_path.exists() \
            else pd.DataFrame()

    def add_record(self, model_name, rmse_mean, rmse_std, featureset_version,
                   correlation_with_target=None, n_folds=None, random_seed=None,
                   delete_old_records=True, round_digits=4, sort_by_rmse_mean=True):

        if delete_old_records and not self.experiment_records.empty:
            self.experiment_records = self.experiment_records[self.experiment_records['model_name'] != model_name]

        if n_folds is None:
            n_folds = self.settings.l2_n_folds

        if random_seed is None:
            random_seed = self.settings.global_seed

        new_record = pd.DataFrame({
            'model_name': [model_name],
            'rmse_mean': [np.round(rmse_mean, round_digits)],
            'rmse_std': [np.round(rmse_std, round_digits)],
            'corr_with_y': [np.round(correlation_with_target, round_digits)],
            'featureset_ver': [featureset_version],
            'num_folds': [n_folds],
            'random_seed': [random_seed],
            'black_listed': [True if model_name in self.settings.black_listed_models else False],
            'timestamp': [pd.Timestamp.now()],
        })

        self.experiment_records = pd.concat([self.experiment_records, new_record], axis=0, ignore_index=True)
        if sort_by_rmse_mean:
            self.experiment_records = self.experiment_records.sort_values(by='rmse_mean', ascending=True)
        else:
            self.experiment_records = self.experiment_records.sort_values(by='timestamp', ascending=False)

        self.experiment_records.to_csv(self.experiment_records_file_path, index=False)
