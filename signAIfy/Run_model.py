import tensorflow as tf
import os
import pandas as pd
import json
import base64
import numpy as np
import logging
import requests
import gzip
import time
from gru import CustomModel as CustomModel3

logging.getLogger().setLevel(logging.ERROR)


df = pd.DataFrame()
file_path = '/path/data/data.csv'

# Check if the file exists
if os.path.exists(file_path):
    df = pd.read_csv("/path/data/data.csv")
else:
    print("file doesnt exist")

train = pd.read_csv("train.csv")
dict_sign = {}
for i, sign in enumerate(train["sign"].unique()):
    dict_sign[i] = sign
batch_size = 1
timesteps = 100
# features = 1086
features = 1629
nb_classes = 250
model = CustomModel3(batch_size, timesteps, features, nb_classes)
# first call to initialize the model.
model(tf.zeros((batch_size, timesteps, features)))
model.load_weights("model1.h5")

def load_parquet_file(df):
    nbframes = len(df["frame"].unique())
    frames = np.zeros((nbframes, 543, 3))
    for step, (name, timestep) in enumerate(df.groupby("frame")):
        frames[step, :, :] = timestep[["x", "y", "z"]].values
    sequence = np.reshape(np.stack(frames), (len(frames), 1629))
    return sequence


seq = load_parquet_file(df)
# split seq in sub seq of 100 timesteps with a paddind of 0 at the end of the sequence
seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1]))
seqs = []
if seq.shape[1] < timesteps:

        seqs.append(
            np.reshape(
                np.concatenate(
                    (seq, np.zeros((1, timesteps - (seq.shape[1]), features))), axis=1
                ), (1, timesteps, features)))

        #seqs.append(np.reshape(seq[0, i:i + timesteps], (1, timesteps, features)))
else:
    zero = np.zeros((1, timesteps - (seq.shape[1] % timesteps), features))
    seq = np.concatenate((seq, zero), axis=1)
    for i in range(0, seq.shape[1], timesteps):
        seqs.append(np.reshape(seq[0, i:i + timesteps], (1, timesteps, features)))
# compute the prediction for each sub seq
for ses_ in seqs:
    p = model(ses_)
    print("prob: ", np.max(p))
    print("classe", dict_sign[np.argmax(p)])

