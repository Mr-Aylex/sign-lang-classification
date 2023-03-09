import pandas as pd
import numpy as np
import os
from tqdm import tqdm

tqdm.pandas()


def load_file(path):
    """

    :param path:
    :return:
    """
    return pd.read_parquet(path, engine='pyarrow')


def load_data(folder_path, train_file):
    """

    :param path:
    :param train_path:
    :param sign:
    :return:
    """

    train = pd.read_csv(os.path.join(folder_path, train_file))

    signs = train['sign'].unique()
    for s in signs:
        res = train[train['sign'] == s]['path'].progress_apply(lambda x: load_file(os.path.join(folder_path, x)))
        lst = []
        for el in res:
            lst.append(el)
        conc = pd.concat(lst)

        conc.to_parquet(f"data/{s}.parquet", engine='pyarrow')


    #train['data'] = train['path'].apply(lambda x: load_file(os.path.join(folder_path, x)))

    return res


