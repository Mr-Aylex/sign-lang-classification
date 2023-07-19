from pathlib import Path

import tensorflow as tf
import keras
from keras import utils, optimizers, losses, metrics
import numpy as np
from keras import callbacks
from keras.layers import TimeDistributed
from keras import layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import keras_nlp
import os
import json
import pandas as pd
from custum_translate_model.first import VarTranslator

# %%

tqdm.pandas()

# %%

DATA_FOLDER = "/mnt/e/sign-lang-data-2"
test = pd.read_csv(os.path.join(DATA_FOLDER, "supplemental_metadata.csv"))

test["npy_path"] = test[["sequence_id", "path"]].progress_apply(
    lambda x: os.path.join(DATA_FOLDER, "npy_train_landmarks2", x["path"].replace(".parquet", "").replace("/", "_"),
                           str(x["sequence_id"]) + ".npy"),
    axis=1
)
test = test[
    test["npy_path"] != "/mnt/e/sign-lang-data-2/npy_train_landmarks2/supplemental_landmarks_1249944812/435344989.npy"]

with open("monBytePairTokenizer.json", "r") as f:
    js_file = json.load(f)

vocab = js_file["model"]["vocab"]
merge = js_file["model"]["merges"]

tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocab, merge)


def load_npy(npy_path, phrase):
    max_nb_frames = 900

    filename = npy_path.numpy().decode("utf-8")
    try:
        inputs = np.load(filename)
    except:
        print(filename)
        raise Exception("error")
    # fill nan with 0
    inputs = np.nan_to_num(inputs)
    if inputs.shape[0] > max_nb_frames:
        inputs = inputs[:max_nb_frames, :, :]
    else:
        inputs = np.concatenate([inputs, np.zeros((max_nb_frames - inputs.shape[0], 543, 3))], axis=0)

    return inputs, phrase.numpy().decode("utf-8")


def add_mask(inputs, label):
    label = label.numpy()
    shape = label.shape
    new_label = np.full((64), 4)
    new_label[:shape[1]] = label[0]
    return inputs, new_label


def load_dataset(dataset):
    # load only the npy
    dataset = dataset.map(
        lambda npy_path, phrase: tf.py_function(func=load_npy, inp=[npy_path, phrase], Tout=(tf.float32, tf.string)),
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(
        lambda inputs, label: (inputs, tokenizer.tokenize(label)),
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(
        lambda inputs, label: tf.py_function(func=add_mask, inp=[inputs, label], Tout=(tf.float32, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    # reshape the input
    dataset = dataset.map(
        lambda inputs, label: (tf.reshape(inputs, (900, 543, 3)), label),
        num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


# %%

# load model

model = VarTranslator(20, vocab, merge, 20)
model(np.zeros((1, 900, 543, 3)))

model.load_weights("checkpoint/VarAutoEncoder/model_4.h5")

# evaluate on test set

test_dataset = tf.data.Dataset.from_tensor_slices((test["npy_path"], test["phrase"]))
test_dataset = load_dataset(test_dataset)

# %%

batch_size = 1

test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# %%

# for inputs, label in tqdm(test_dataset):
#    pass


for inputs, label in tqdm(test_dataset):
    pred = model.predict(inputs)
    print(pred.shape)
    pred = np.argmax(pred, axis=-1)
    print(pred.shape)
    pred = pred.astype(dtype=np.int32)

    print(pred)
    print(tokenizer.detokenize(pred))
    print(tokenizer.detokenize(label))

# %%
#model.compile(loss=losses.SparseCategoricalCrossentropy(),
#              metrics=[metrics.SparseCategoricalAccuracy()])

#model.evaluate(test_dataset)

# %%


