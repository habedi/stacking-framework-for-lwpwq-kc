import os
from pathlib import Path

from l0_settings import Settings

global_settings = Settings()

import pandas as pd


def remove_columns_with_constant_values(df):
    df_copy = df.copy()
    constant_features = [feature for feature in df_copy.columns if df_copy[feature].nunique() == 1]
    print(f"Removing {len(constant_features)} constant features: {constant_features}")
    df_copy.drop(constant_features, axis=1, inplace=True)
    return df_copy


# Must work with float features too
def remove_columns_with_almost_constant_values(df, threshold=0.002):
    df_copy = df.copy()
    num_rows = df_copy.shape[0]
    features_with_almost_constant_values = [feature for feature in df_copy.columns
                                            if df_copy[feature].nunique() / num_rows < threshold]
    print(f"Removing {len(features_with_almost_constant_values)} features with almost constant values: "
          f"{features_with_almost_constant_values}")
    df_copy.drop(features_with_almost_constant_values, axis=1, inplace=True)
    return df_copy


def remove_features_with_too_many_missing_values(df, max_num_missing_values=global_settings.max_missing_values_ratio):
    df_copy = df.copy()
    num_rows = df_copy.shape[0]
    features_with_too_many_missing_values = [feature for feature in df_copy.columns
                                             if df_copy[feature].isnull().sum() / num_rows > max_num_missing_values]
    df_copy.drop(features_with_too_many_missing_values, axis=1, inplace=True)
    return df_copy


def remove_categorical_features_with_high_cardinality(df, max_unique_values=global_settings.max_num_unique_values):
    df_copy = df.copy()
    categorical_features = df_copy.select_dtypes(include=['object', 'int']).columns
    categorical_features_with_high_cardinality = [feature for feature in categorical_features
                                                  if df_copy[feature].nunique() > max_unique_values]
    df_copy.drop(categorical_features_with_high_cardinality, axis=1, inplace=True)
    return df_copy


num_featuresets = 0
for file in Path(os.path.dirname(__file__)).glob('l1_prepare_raw_features_*.py'):
    num_featuresets += 1

for v in range(1, num_featuresets + 1):
    print(f"\nReading featureset {v}...")

    raw_train_features_with_scores = pd.read_csv(global_settings.raw_features_dir /
                                                 f'raw_train_features_with_scores_v{v}.csv.gz',
                                                 index_col=global_settings.id_col)

    raw_train_features = raw_train_features_with_scores.drop(global_settings.target_col, axis=1)
    scores = raw_train_features_with_scores[global_settings.target_col]

    print(f"raw_train_features.shape: {raw_train_features.shape}")

    raw_test_features = pd.read_csv(global_settings.raw_features_dir /
                                    f'raw_test_features_v{v}.csv.gz',
                                    index_col=global_settings.id_col)

    print(f"raw_test_features.shape: {raw_test_features.shape}")

    filtered_train_features = remove_features_with_too_many_missing_values(raw_train_features)

    print(f"filtered_train_features.shape: {filtered_train_features.shape} "
          f"after removing features with too many missing values")

    # filtered_train_features = remove_categorical_features_with_high_cardinality(filtered_train_features)

    # filtered_train_features = remove_columns_with_almost_constant_values(filtered_train_features)
    # print(f"filtered_train_features.shape: {filtered_train_features.shape} "
    #       f"after removing constant features")

    filtered_train_features_with_scores = pd.concat([filtered_train_features, scores], axis=1)

    filtered_test_features = raw_test_features.loc[:, filtered_train_features.columns]
    print(f"filtered_test_features.shape: {filtered_test_features.shape}")

    filtered_train_features_with_scores.to_csv(
        global_settings.raw_features_dir / f'train_features_with_scores_v{v}.csv.gz', index=True)
    filtered_test_features.to_csv(global_settings.raw_features_dir / f'test_features_v{v}.csv.gz', index=True)
