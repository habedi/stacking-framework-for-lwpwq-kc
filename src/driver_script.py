import os

import pandas as pd

from l0_settings import Settings

global_settings = Settings()

# # Performing some tidying up

# In[2]:


os.system('rm -rf oofs catboost_info submission.csv.gz meta_models_oofs experiment_records.csv.gz')

# # Check the list of models in the stacking pipeline

# In[3]:


os.system('python3 l0_settings.py')

# # Make the (raw) features used by the models

# In[4]:


if global_settings.rebuild_features:
    os.system('echo "Performing feature engineering"')
    os.system('rm -rf raw_features')
    # os.system('python l0_fix_encoding.py')
    os.system('python l1_prepare_raw_features_v1.py')
    os.system('python l1_prepare_raw_features_v2.py')
    os.system('python l1_prepare_raw_features_v3.py')

# # Performing feature selection

# In[5]:


os.system('echo "Performing feature selection on raw features"')
os.system('python l1_filter_features.py')

# # Run the script associated with each model

# In[6]:


for file in global_settings.output_dir.glob("l2_*.py"):
    if file.stem in global_settings.black_listed_models:
        print(f"Skipping {file.stem} as it is blacklisted")
    else:
        print(f"Running {file}")
        os.system(f'python {file}')
        print(f"Finished running {file}")
        print(50 * "-")

# # Run the script for training the metamodel and performing inference on test data

# In[7]:


os.system('echo "Running layer 3 models"')
os.system('python l3_meta_model_lgbm.py')
os.system('python l3_meta_model_huber.py')
os.system('python l3_meta_model_xgboost.py')
os.system('python l3_meta_model_catboost.py')
os.system('python l3_meta_model_tpot.py')

# In[8]:


os.system('echo "Running layer 4 models"')
os.system('python l4_blending_meta_models.py')

# # Have a look at the feature importance

# In[9]:


performances = pd.read_csv(global_settings.output_dir / "experiment_records.csv.gz")
performances.head(20)

# # Playground

# In[9]:
