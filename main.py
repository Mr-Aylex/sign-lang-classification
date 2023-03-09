import pandas as pd
import numpy as np
import os
from tqdm import tqdm

tqdm.pandas()

DATA_FOLDER = "/mnt/d/sign-lang-data/"
if __name__ == '__main__':
    from data_preparation.load_data import load_data
    load_data(DATA_FOLDER, 'train.csv')