"""
This script uses the data from the csv files and creates a numpy array for each video of parquet file.
"""

import numpy as np
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=15, progress_bar=True)
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# %%

tqdm.pandas()
# keras set seed
# Load the data
DATA_FOLDER = "/mnt/e/sign-lang-data/"
train = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv"))

if not os.path.exists(os.path.join(DATA_FOLDER, "sequences")):
    os.mkdir(os.path.join(DATA_FOLDER, "sequences"))

# %%
def convert_to_npy(path):
    new_path = path.replace(".parquet", ".npy")
    new_path = new_path.replace("train", "train_npy")
    new_path = new_path.replace("/", "_")

    new_path = os.path.join("sequences", new_path)

    df = pd.read_parquet(os.path.join(DATA_FOLDER, path), engine="pyarrow", columns=["frame", "x", "y", "z"])
    df = df.fillna(0)

    nbframes = len(df["frame"].unique())
    frames = np.zeros((nbframes, 543, 3))
    for step, (name, timestep) in enumerate(df.groupby("frame")):
        frames[step, :, :] = timestep[["x", "y", "z"]].values

    np.save(os.path.join(DATA_FOLDER, new_path), frames)

    return new_path


# %%
train["new_path"] = train["path"].progress_apply(lambda x: convert_to_npy(x))

# %%
train.to_csv(os.path.join(DATA_FOLDER, "train_processed.csv"), index=False)
