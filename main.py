import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from time import time
import pickle
import numpy as np

from src.data.processing_data import processing_features
from src.train.boosting import training_boosting
from src.predict.evaluate_predict import evaluate
from src.utils.save_load import *
# legend:dict, local_path:Path = os.path.join(Path(__file__).parent.parent,"data"))

import os
import mlflow

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("DEV-IRBoosting")

is_Train = False
is_Exp = True

X_mods,X_test,y_train,y_test = processing_features()


with mlflow.start_run():

    mlflow.set_tag("Ambiente","fair-refa-v2")

    if is_Train:
        ir_boost = training_boosting(X_mods,y_train)

        evaluate(ir_boost,X_mods,X_test,y_train,y_test,is_Exp)
        
        save_model(ir_boost)
    else:
        ir_boost = load_model()

        evaluate(ir_boost,X_mods,X_test,y_train,y_test,is_Exp)