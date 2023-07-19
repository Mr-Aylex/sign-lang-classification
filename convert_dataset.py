# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from numba import njit
from tqdm import tqdm
import asyncio

tqdm.pandas()
import concurrent.futures

# %%

MAX_NB_FRAMES = 900
DATA_FOLDER = "/mnt/e/sign-lang-data-2"


# print(all_col)

# %%
# this notebook is for the purpose of testing a data input pipeline
# %%
def load_data(filename):
    # filename = filename.numpy().decode("utf-8")

    max_nb_frames = MAX_NB_FRAMES
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    seq_label = pd.read_csv(os.path.join(DATA_FOLDER, "train", str(filename).replace("/", "_") + ".csv"), sep=";")
    seq_label = seq_label[["sequence_id", "phrase"]]

    parquet = pd.read_parquet(os.path.join(DATA_FOLDER, filename), engine='pyarrow')

    all_col = list(parquet.columns)

    parquet_group = parquet.groupby("sequence_id")

    x_face = [col for col in all_col if "x_face" in col]
    y_face = [col for col in all_col if "y_face" in col]
    z_face = [col for col in all_col if "z_face" in col]

    x_left_hand = [col for col in all_col if "x_left_hand" in col]
    y_left_hand = [col for col in all_col if "y_left_hand" in col]
    z_left_hand = [col for col in all_col if "z_left_hand" in col]

    x_right_hand = [col for col in all_col if "x_right_hand" in col]
    y_right_hand = [col for col in all_col if "y_right_hand" in col]
    z_right_hand = [col for col in all_col if "z_right_hand" in col]

    x_pose = [col for col in all_col if "x_pose" in col]
    y_pose = [col for col in all_col if "y_pose" in col]
    z_pose = [col for col in all_col if "z_pose" in col]

    stored_data = np.zeros((len(parquet_group), max_nb_frames, 543, 3))

    # crerate a tf tensor with shape (len(parquet_group), 1)

    for step, (name, sequence) in enumerate(parquet_group):
        for sub_step, (sub_name, timestep) in enumerate(sequence.groupby("frame")):
            stored_data[step, sub_step, :468, 0] = timestep[x_face].values
            stored_data[step, sub_step, :468, 1] = timestep[y_face].values
            stored_data[step, sub_step, :468, 2] = timestep[z_face].values

            stored_data[step, sub_step, 468:468 + 21, 0] = timestep[x_left_hand].values
            stored_data[step, sub_step, 468:468 + 21, 1] = timestep[y_left_hand].values
            stored_data[step, sub_step, 468:468 + 21, 2] = timestep[z_left_hand].values

            stored_data[step, sub_step, 468 + 21:468 + 21 * 2, 0] = timestep[x_right_hand].values
            stored_data[step, sub_step, 468 + 21:468 + 21 * 2, 1] = timestep[y_right_hand].values
            stored_data[step, sub_step, 468 + 21:468 + 21 * 2, 2] = timestep[z_right_hand].values

            stored_data[step, sub_step, 468 + 21 * 2:468 + 21 * 2 + 33, 0] = timestep[x_pose].values
            stored_data[step, sub_step, 468 + 21 * 2:468 + 21 * 2 + 33, 1] = timestep[y_pose].values
            stored_data[step, sub_step, 468 + 21 * 2:468 + 21 * 2 + 33, 2] = timestep[z_pose].values

    return stored_data, seq_label.values[:, 1]


@njit
def fill_data(stored_data, x_face, y_face, z_face, x_left_hand, y_left_hand, z_left_hand, x_right_hand, y_right_hand,
              z_right_hand, x_pose, y_pose, z_pose, sub_step):
    stored_data[sub_step, :468, 0] = x_face
    stored_data[sub_step, :468, 1] = y_face
    stored_data[sub_step, :468, 2] = z_face

    stored_data[sub_step, 468:468 + 21, 0] = x_left_hand
    stored_data[sub_step, 468:468 + 21, 1] = y_left_hand
    stored_data[sub_step, 468:468 + 21, 2] = z_left_hand

    stored_data[sub_step, 468 + 21:468 + 21 * 2, 0] = x_right_hand
    stored_data[sub_step, 468 + 21:468 + 21 * 2, 1] = y_right_hand
    stored_data[sub_step, 468 + 21:468 + 21 * 2, 2] = z_right_hand

    stored_data[sub_step, 468 + 21 * 2:468 + 21 * 2 + 33, 0] = x_pose
    stored_data[sub_step, 468 + 21 * 2:468 + 21 * 2 + 33, 1] = y_pose
    stored_data[sub_step, 468 + 21 * 2:468 + 21 * 2 + 33, 2] = z_pose


def create_npy(filename):
    # filename = filename.numpy().decode("utf-8")

    max_nb_frames = MAX_NB_FRAMES
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import time
    from pathlib import Path

    start = time.time()
    parquet = pd.read_parquet(os.path.join(DATA_FOLDER, filename), engine='pyarrow')
    end = time.time()
    # print("time to read parquet file : ", end - start)

    all_col = list(parquet.columns)

    start = time.time()
    parquet_group = parquet.groupby("sequence_id")
    end = time.time()
    # print("time to group parquet file : ", end - start)

    x_face = [col for col in all_col if "x_face" in col]
    y_face = [col for col in all_col if "y_face" in col]
    z_face = [col for col in all_col if "z_face" in col]

    x_left_hand = [col for col in all_col if "x_left_hand" in col]
    y_left_hand = [col for col in all_col if "y_left_hand" in col]
    z_left_hand = [col for col in all_col if "z_left_hand" in col]

    x_right_hand = [col for col in all_col if "x_right_hand" in col]
    y_right_hand = [col for col in all_col if "y_right_hand" in col]
    z_right_hand = [col for col in all_col if "z_right_hand" in col]

    x_pose = [col for col in all_col if "x_pose" in col]
    y_pose = [col for col in all_col if "y_pose" in col]
    z_pose = [col for col in all_col if "z_pose" in col]

    # stored_data = np.zeros((len(parquet_group),max_nb_frames, 543, 3))

    # crerate a tf tensor with shape (len(parquet_group), 1)

    async def process_group(name_sequence_pair):
        name, sequence = name_sequence_pair
        if os.path.exists(os.path.join(DATA_FOLDER, "npy_train_landmarks2",
                                       str(filename).replace("/", "_").replace(".parquet", ""),
                                       str(name).replace("/", "_") + ".npy")):
            return None, None
        stored_data = np.zeros((max_nb_frames, 543, 3))
        max_frame = 0
        for sub_step, (sub_name, timestep) in enumerate(sequence.groupby("frame")):
            fill_data(
                stored_data,
                timestep[x_face].values, timestep[y_face].values, timestep[z_face].values,
                timestep[x_left_hand].values, timestep[y_left_hand].values, timestep[z_left_hand].values,
                timestep[x_right_hand].values, timestep[y_right_hand].values, timestep[z_right_hand].values,
                timestep[x_pose].values, timestep[y_pose].values, timestep[z_pose].values,
                sub_step
            )
            max_frame = sub_step
        Path(os.path.join(DATA_FOLDER, "npy_train_landmarks2",
                          str(filename).replace("/", "_").replace(".parquet", ""))).mkdir(exist_ok=True)

        np.save(
            os.path.join(DATA_FOLDER, "npy_train_landmarks2", str(filename).replace("/", "_").replace(".parquet", ""),
                         str(name).replace("/", "_") + ".npy"), stored_data[:max_frame + 1, :, :]
        )

    async def main():
        await asyncio.gather(*[process_group(name_sequence_pair) for name_sequence_pair in parquet_group])

    asyncio.run(main())

    # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #    # for step, group in tqdm(enumerate(parquet_group)):
    #    list(executor.map(process_group, parquet_group))


train = pd.read_csv(os.path.join(DATA_FOLDER, "supplemental_metadata.csv"))

# train = train[train["sequence_id"] == 383831158]
print(train)
train["path"].drop_duplicates().progress_apply(create_npy)
