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
logging.error("---------------------------------------------------------------------------")

logging.getLogger().setLevel(logging.ERROR)

url = 'http://localhost:5000/dataframe'
df = pd.DataFrame()

response = requests.get(url)
logging.error(response.status_code == 200)
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame.from_dict(data)

logging.error("afterloop")
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
    print(df.head())
    nbframes = len(df["frame"].unique())
    frames = np.zeros((nbframes, 543, 3))
    for step, (name, timestep) in enumerate(df.groupby("frame")):
        frames[step, :, :] = timestep[["x", "y", "z"]].values
    print(frames.shape)
    sequence = np.reshape(np.stack(frames), (len(frames), 1629))
    return sequence


seq = load_parquet_file(df)
# split seq in sub seq of 100 timesteps with a paddind of 0 at the end of the sequence
seqs = []
if seq.shape[1] < timesteps:
    for i in range(0, seq.shape[1], timesteps):
        if i + timesteps > seq.shape[1]:
            seqs.append(np.reshape(np.concatenate((seq[0, i:], np.zeros((timesteps - (seq.shape[1] - i), features)))),
                                   (1, timesteps, features)))
        else:
            seqs.append(np.reshape(seq[0, i:i + timesteps], (1, timesteps, features)))
else:
    seq = np.concatenate((seq, np.zeros((timesteps - len(seq), 1629))))
    seqs.append(np.reshape(seq, (1, timesteps, features)))
# compute the prediction for each sub seq
print("seqs: ", len(seqs))
print("seqs[0]: ", seqs[0].shape)
for ses_ in seqs:
    p = model(ses_)
    print("prob: ", np.max(p))
    print("classe", dict_sign[np.argmax(p)])

print("hello")