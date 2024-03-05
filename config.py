from typing import Union, Any

import pandas as pd
import numpy as np


DATA_FOLDER = "data"
DATA_PCL_NAME = "dontpatronizeme_pcl.csv"
DATA_CATEGORIES_NAME = "dontpatronizeme_categories.csv"
TRAIN_ID = "train_semeval_parids-labels.csv"
DEV_ID = "dev_semeval_parids-labels.csv"

#BASELINE_DF_NAME = "df_baseline.csv"
#MODEL_FOLDER = "models"
#MODEL_NAME = "nlp_model.pt"
#BASELINE_PATH = "outputs/"
#PLOT_FOLDER = "plots"
# Type
Array_like = Union[list, np.ndarray, pd.Series, pd.DataFrame, Any]

# Model type
MODEL_TYPE = ["roberta", "roberta_base"]
ARGS = classification_arg
NUM_LABELS = 2
USE_CUDA = "cuda_available"
  
# Hyperparameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.001
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 100

# Model arguments
NUM_TRAIN_EPOCHS = 1
NO_SAVE = "True"
NO_CACHE = "True"
OVERWRITE_OUTPUT_DIR = "True"
     
# Paths
save_model_path = "/path/to/save_model"
data1_path = /path/to/data1
data2_path = /path/to/data2
images_path = /path/to/images
logs_path = /path/to/logs
