import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from custom_model.gru import CustomModel as CustomModel3

# %%
DATA_FOLDER = "/mnt/e/sign-lang-data/"
train = pd.read_csv(os.path.join(DATA_FOLDER, "train_processed.csv"))
dict_sign = {}
for i, sign in enumerate(train["sign"].unique()):
    dict_sign[i] = sign
batch_size = 1
timesteps = 100
# features = 1086
features = 1629
nb_classes = 250

# %%
# load model and weights
model = CustomModel3(batch_size, timesteps, features, nb_classes)
# first call to initialize the model.
model(tf.zeros((batch_size, timesteps, features)))
model.load_weights("checkpoint/gru/model1.h5")


# define the sequence

def load_parquet_file(filename):
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(filename, engine="pyarrow", columns=["frame", "x", "y", "z"])
    df = df.fillna(0)

    # npy_data = np.load(filename.numpy()).astype(np.float32)
    # Return the data and label as a tuple
    # convert label custom_model numpy array
    # convert label (b'[0,1]') to [0,1]

    nbframes = len(df["frame"].unique())
    frames = np.zeros((nbframes, 543, 3))
    for step, (name, timestep) in enumerate(df.groupby("frame")):
        frames[step, :, :] = timestep[["x", "y", "z"]].values
    print(frames.shape)
    sequence = np.reshape(np.stack(frames), (len(frames), 1629))
    return sequence

seq = load_parquet_file(f"{DATA_FOLDER}/train_landmark_files/26734/3829304.parquet")
print(train[train["path"] == "train_landmark_files/26734/3829304.parquet"]["sign"])
#seq = np.zeros((1, 259, 1629))  # shape (1, timesteps, 1629)
print(seq.shape)
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

# %%
