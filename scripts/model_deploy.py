# %%
#1. Import packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os,pickle,re

#2. Functions to load the scaler, oversampler and campaign_outcome_model
def load_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        pickle_object = pickle.load(f)
    return pickle_object

def load_model(filepath):
    model_loaded = keras.models.load_model(filepath)
    return model_loaded

#3.Define the file paths towards the scaler, oversampler and the model
PATH = os.getcwd()
oversampler_filepath = os.path.join(PATH, "oversampler.pkl")
scaler_filepath = os.path.join(PATH, "scaler.pkl")
model_filepath = os.path.join(PATH, "campaign_outcome_model")

#4. Load the oversampler, model and scaler
oversampler = load_pickle_file(oversampler_filepath)
scaler = load_pickle_file(scaler_filepath)
model = load_model(model_filepath)
# %%
