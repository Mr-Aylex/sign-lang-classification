import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sn
import tensorflow as tf
import tensorflow_addons as tfa

from tqdm import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from leven import levenshtein

import glob
import sys
import os
import math
import gc
import sys
import sklearn
import time
import json

from custom_model.encoder_generator.utils import *
from custom_model.encoder_generator.model import *

# %%
DATA_FOLDER = "/mnt/e/sign-lang-data-2"
train = pd.read_csv(f'{DATA_FOLDER}/train.csv')
MODEL_NAME = 'encoder_generator_96_64'
convert_to_tf_lite = True
# %%
N_WARMUP_EPOCHS = 10
LR_MAX = 1e-3
N_EPOCHS = 100
#
# MatplotLib Global Settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 24
# %%
train['file_path'] = train['path'].apply(get_file_path)

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')[:, :MAX_PHRASE_LENGTH]

n_train_sample = len(X_train)
# VAL
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')[:, :MAX_PHRASE_LENGTH]
n_val_sample = len(X_val)
# %%
# load mean and std for normalization
MEANS = np.load('data/MEANS.npy').reshape(-1)
STDS = np.load('data/STDS.npy').reshape(-1)
# %%
batch_size = 128
preprocess_layer = PreprocessLayer()
train_dataset = get_train_dataset(X_train, y_train, batch_size)
val_dataset = get_val_dataset(X_val, y_val, batch_size)
# %%
params = {
    'n_frames': 128,  # don't touch
    'n_cols': 164,  # don't touch
    'max_phrase_length': MAX_PHRASE_LENGTH,
    'n_unique_characters': N_UNIQUE_CHARACTERS,
    'num_blocks_encoder': 3,
    'num_blocks_decoder': 3,
    'units_encoder': 96,  # default 384
    'units_decoder': 64,  # default 256
    'num_head': 4,
    'mlp_ratio': 2,
    'mh_dropout_ratio': 0.3,
    'layer_nom_eps': 1e-6,
    'mlp_dropout_ratio': 0.30,
    'classifier_dropout_ratio': 0.10,
}
# %%
model = get_model(mean=MEANS, std=STDS, params=params)

print(model.summary(expand_nested=True))
tf.keras.utils.plot_model(
    model, to_file=f'{MODEL_NAME}.png',
    show_shapes=True, show_dtype=True,
    show_layer_names=True, expand_nested=True,
    show_layer_activations=True
)
# %%

n_val_step_per_epoch = math.ceil(n_val_sample / batch_size)
verify_no_nan_predictions(model, val_dataset, n_val_step_per_epoch)
# %%
LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_training_steps=N_EPOCHS, num_cycles=0.50)
               for step in range(N_EPOCHS)]
# Plot Learning Rate Schedule
plot_lr_schedule(LR_SCHEDULE, epochs=N_EPOCHS)

y_pred = model.evaluate(
    val_dataset,
    steps=n_val_step_per_epoch
)
baseline_accuracy = np.mean(y_val == PAD_TOKEN)
# %%
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)

csv_logger = tf.keras.callbacks.CSVLogger(f'log_{MODEL_NAME}.csv')

train_step_per_epoch = math.ceil(n_train_sample / batch_size)
# Actual Training
history = model.fit(
    x=train_dataset,
    steps_per_epoch=train_step_per_epoch,
    epochs=N_EPOCHS,
    # Only used for validation data since training data is a generator
    validation_data=val_dataset,
    validation_steps=n_val_step_per_epoch,
    callbacks=[
        lr_callback,
        WeightDecayCallback(),
        csv_logger
    ]
)
model.save(f'{MODEL_NAME}.h5')
# %%
model.evaluate(
    val_dataset,
    steps=n_val_step_per_epoch,
    batch_size=batch_size
)
# %%
LD_TRAIN_DF = get_ld_train(x_train=X_train, y_train=y_train, model=model)
LD_VAL_DF = get_ld_val(x_val=X_val, y_val=y_val, model=model)

# %%
# show the distribution of Levenstein Distance
LD_TRAIN_VC = dict([(i, 0) for i in range(LD_TRAIN_DF['levenshtein_distance'].max() + 1)])
for ld in LD_TRAIN_DF['levenshtein_distance']:
    LD_TRAIN_VC[ld] += 1

plt.figure(figsize=(15, 8))
pd.Series(LD_TRAIN_VC).plot(kind='bar', width=1)
plt.title(f'Train Levenstein Distance Distribution | Mean: {LD_TRAIN_DF.levenshtein_distance.mean():.4f}')
plt.xlabel('Levenstein Distance')
plt.ylabel('Sample Count')
plt.xlim(-0.50, LD_TRAIN_DF.levenshtein_distance.max() + 0.50)
plt.grid(axis='y')
plt.savefig(f'{MODEL_NAME}_train_levenshtein_distance_distribution.png')
plt.show()

LD_VAL_VC = dict([(i, 0) for i in range(LD_VAL_DF['levenshtein_distance'].max() + 1)])
for ld in LD_VAL_DF['levenshtein_distance']:
    LD_VAL_VC[ld] += 1

plt.figure(figsize=(15, 8))
pd.Series(LD_VAL_VC).plot(kind='bar', width=1)
plt.title(f'Validation Levenstein Distance Distribution | Mean: {LD_VAL_DF.levenshtein_distance.mean():.4f}')
plt.xlabel('Levenstein Distance')
plt.ylabel('Sample Count')
plt.xlim(0 - 0.50, LD_VAL_DF.levenshtein_distance.max() + 0.50)
plt.grid(axis='y')
plt.savefig(f'{MODEL_NAME}_validation_levenshtein_distance_distribution.png')
plt.show()

plot_history_metric(history, 'loss', MODEL_NAME, f_best=np.argmin)

plot_history_metric(history, 'top1acc', MODEL_NAME, ylim=[0, 1], yticks=np.arange(0.0, 1.1, 0.1))

plot_history_metric(history, 'top5acc', MODEL_NAME, ylim=[0, 1], yticks=np.arange(0.0, 1.1, 0.1))

if convert_to_tf_lite:
    # Define TF Lite Model
    tflite_keras_model = TFLiteModel(model)

    # Create Model Converter
    keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
    # Convert Model
    tflite_model = keras_model_converter.convert()
    # Write Model
    with open(f'model_save/{MODEL_NAME}.tflite', 'wb') as f:
        f.write(tflite_model)

    
