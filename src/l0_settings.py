from pathlib import Path


class Settings:
    name = 'global_settings'
    global_seed = 42
    l2_n_folds = 10
    l3_n_folds = 5
    target_col = 'score'
    id_col = 'id'
    rebuild_features = False
    use_stratified_folds = True
    max_num_unique_values = 3000
    max_missing_values_ratio = 0.05
    experiment_records_file_name = 'experiment_records.csv.gz'
    blending_cutoff = 4
    black_listed_models = []
    # Seeds for meta-models' training and inference
    random_seed_offsets = [0, 0, 0, 0, 0]

    def __init__(self, featureset_version=3, is_in_kaggle=False):
        # File paths
        if is_in_kaggle:
            data_dir = Path('./linking-writing-processes-to-writing-quality')
            output_dir = Path('./')
        else:
            data_dir = Path('../data/competition_data')
            output_dir = Path('./')

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.oof_dir = output_dir / 'oofs'
        self.raw_features_dir = output_dir / 'raw_features'
        self.featureset_version = featureset_version
        self.meta_model_oofs_dir = output_dir / 'meta_models_oofs'

    def get_stacking_models_list(self):
        return [script.stem for script in Path(__file__).parent.glob('l2*') if script.is_file()]

    def get_train_logs_path(self, file_name='train_logs.csv.gz'):
        return self.data_dir / file_name

    def get_train_scores_path(self, file_name='train_scores.csv.gz'):
        return self.data_dir / file_name

    def get_test_logs_path(self, file_name='test_logs.csv.gz'):
        return self.data_dir / file_name

    def get_featureset_version(self):
        return self.featureset_version

    def get_black_listed_meta_features(self):
        return ["oof_" + i for i in self.black_listed_models]

    def get_meta_model_oofs_dir(self):
        if not self.meta_model_oofs_dir.exists():
            self.meta_model_oofs_dir.mkdir(parents=True)
        return self.meta_model_oofs_dir


if __name__ == '__main__':
    global_settings = Settings()
    print(global_settings.get_stacking_models_list())
