# Stacking Framework

This repository contains the code for a stacking framework designed for
the [Linking Writing Processes to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality)
Kaggle competition.

![Framework overview](static/framework_overview.drawio.png)

## Folder Structure

The folder structure of the framework is as follows:

- `bin`: This folder includes the binary file for [DuckDB](https://duckdb.org/).
- `data`: This folder includes
  the [competition data](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/data).
- `src`: This folder includes the framework's source code.

## Framework Layers

The framework consists of five sequential layers, each needing the previous layer's output as input.

### Layer 0

The first layer includes a few Python scripts that primarily implement logging routines and define global settings for
the framework. The names of the scripts in this layer start with `l0_`, such as `l0_settings.py`.

### Layer 1

The second layer consists of Python scripts that mainly implement data preprocessing, feature engineering, and simple
feature selection routines. The names of the scripts in this layer start with `l1_`, such as `l1_filter_features.py`.

### Layer 2

The third layer includes the code for the base models used for stacking. Each Python file or script in this layer
implements one base model. The names of the scripts in this layer start with `l2_`, such as `l2_base_model_xgboost.py`.

### Layer 3

The fourth layer includes the code for the meta-models or the stacking models. Each Python file or script in this layer
implements one meta-model. The names of the scripts in this layer start with `l3_`, such as `l3_meta_model_xgboost.py`.

### Layer 4

The fifth layer includes the code for an ensemble model using simple blending. The Python script in this layer
implements blending by computing a weighted average for the predictions of the meta-models from the previous layer. The
name of the script(s) in this layer starts with `l4_`, such as `l4_blending_meta_models.py`.

## Installing Dependencies

To install the dependencies, you need to have [Poetry](https://python-poetry.org/) installed. You can install Poetry via
Pip using the following command:

`pip install poetry`

To initiate the Poetry environment and install the dependency packages, run the following commands in the shell in the
root folder of this repository after downloading it.

`poetry update && poetry init`

After that, enter the Poetry environment by invoking the poetry's shell using the following command:

`poetry shell`

## Running the Framework End-to-End ML Pipeline

To run the entire framework as an end-to-end pipeline, execute the `driver_script.py`.

## Important Output Files

The framework generates the following two main output files after running successfully as a pipeline:

- `src/submission.csv`: This is the final submission file for the competition.
- `src/experiment_records.csv`: This file contains information about the performance of the models in the framework,
  including the performance of the base and meta/stacking models.

## Using DuckDB

You can use DuckDB to work with the CSV files and see the performance of the models, which is recorded in
the `experiment_records.csv` file. See the picture below for an example.

![](static/model_performance.png)

## Licenses

### Data

The data stored here (like the processed CSV files) are licensed
under [Creative Commons license](http://creativecommons.org/licenses/by/4.0/).
Please
visit [competition's webpage](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/data) for
licenses that apply to the original competition data.

### Code

The code in this repository is available under [Apache License](LICENSE).
