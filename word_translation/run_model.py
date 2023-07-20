import tensorflow as tf
import keras
from keras import utils, optimizers, losses, metrics
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt

# %%

tqdm.pandas()
# keras set seed
utils.set_random_seed(1234)
# Load the data
DATA_FOLDER = "/mnt/e/sign-lang-data/"
train = pd.read_csv(os.path.join(DATA_FOLDER, "train_processed.csv"))
dict_sign = {}
for i, sign in enumerate(train["sign"].unique()):
    dict_sign[sign] = i
# %%

from custom_model.lstm import CustomModel
from custom_model.simple_rnn import CustomModel as CustomModel2
from custom_model.gru import CustomModel as CustomModel3

batch_size = 64
timesteps = 100
features = 1086
# features = 1629
nb_classes = len(train["sign"].unique())

model = CustomModel(batch_size, timesteps, features, nb_classes)
# first call to initialize the model.
model(tf.zeros((batch_size, timesteps, features)))
#model.load_weights("checkpoint/lstm/64/model95.h5")


# model = tf.keras.saving.load_model("checkpoint/gru2/model.h5")


# model = CustomModel( input_layer, output_layer)
# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode='temporal')
# %%
def get_label(sign, dict_):
    return utils.to_categorical(dict_[sign], num_classes=len(train["sign"].unique()))


# %%
train["new_path"] = train["new_path"].progress_apply(lambda x: os.path.join(DATA_FOLDER, x))
# train["label"] = train["sign"].parallel_apply(lambda x: get_label(x, dict_sign))
# %%
X = train["new_path"].values
Y = train["sign"].values
# %%
# split the dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=42)
# %%
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# %%

def load_npy_file(filename, label):
    # Load the NPY file
    max_nb_frames = timesteps
    import numpy as np

    filename = filename.numpy().decode("utf-8")
    label = label.numpy().decode("utf-8")

    sequence = np.load(filename).astype(np.float32)
    # get only the x and y coordinates
    sequence = sequence[:, :, :2]

    sequence = np.reshape(sequence, (sequence.shape[0], 1086))
    # sequence = np.reshape(sequence, (sequence.shape[0], 1086))

    if len(sequence) > max_nb_frames:
        sequence = sequence[:max_nb_frames]
    else:
        sequence = np.concatenate((sequence, np.zeros((max_nb_frames - len(sequence), 1086))))
        # sequence = np.concatenate((sequence, np.zeros((max_nb_frames - len(sequence), 1086))))
    label = utils.to_categorical(dict_sign[label], num_classes=250)
    return sequence, label


def load_parquet_file(filename, label):
    max_nb_frames = timesteps
    import numpy as np
    import pandas as pd

    filename = filename.numpy().decode("utf-8")
    label = label.numpy().decode("utf-8")

    df = pd.read_parquet(filename, engine="pyarrow", columns=["frame", "x", "y"])
    df = df.fillna(0)

    # npy_data = np.load(filename.numpy()).astype(np.float32)
    # Return the data and label as a tuple
    # convert label custom_model numpy array
    # convert label (b'[0,1]') to [0,1]

    nbframes = len(df["frame"].unique())
    frames = np.zeros((nbframes, 543, 2))
    for step, (name, timestep) in enumerate(df.groupby("frame")):
        frames[step, :, :] = timestep[["x", "y"]].values

    sequence = np.reshape(np.stack(frames), (len(frames), 1086))

    if len(sequence) > max_nb_frames:
        sequence = sequence[:max_nb_frames]
    else:
        sequence = np.concatenate((sequence, np.zeros((max_nb_frames - len(sequence), 1086))))
    label = utils.to_categorical(dict_sign[label], num_classes=250)
    return sequence, label


# %%
# Shuffle the dataset
train_dataset = train_dataset.shuffle(buffer_size=train["path"].shape[0])

# Load the NPY files and labels in batches
train_dataset = train_dataset.map(
    lambda filename, label: tf.py_function(
        load_npy_file, [filename, label], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
# Prefetch the data for improved performance
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
# %%
# Shuffle the dataset
val_dataset = val_dataset.shuffle(buffer_size=train["path"].shape[0])

# Load the NPY files and labels in batches
val_dataset = val_dataset.map(
    lambda filename, label: tf.py_function(
        load_npy_file, [filename, label], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
# Prefetch the data for improved performance
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
# %%
optimizer = optimizers.Adam(learning_rate=0.0009)  # optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)

# Instantiate a loss function.
loss_fn = losses.CategoricalCrossentropy()

# Prepare the metrics.
train_acc_metric = metrics.CategoricalAccuracy()
val_acc_metric = metrics.CategoricalAccuracy()
# %%
import time


@tf.function
def train_step(model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric, max_sequence_length=150):
    for seq in range(timesteps // max_sequence_length):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train[:, seq:seq + max_sequence_length], training=True)
            loss_value = loss_fn(y_batch_train[:, :], logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    model.reset_states()
    # Update training metric.
    train_acc_metric.update_state(y_batch_train[:, :], logits)

    return loss_value


@tf.function
def test_step(model, x_batch_train, y_batch_train, loss_fn, train_acc_metric, max_sequence_length=150):
    for seq in range(timesteps // max_sequence_length):
        logits = model(x_batch_train[:, seq:seq + max_sequence_length], training=True)
        loss_value = loss_fn(y_batch_train[:, :], logits)
    model.reset_states()
    # Update training metric.
    train_acc_metric.update_state(y_batch_train[:, :], logits)

    return loss_value


def custom_fit(model, epochs, train_dataset, val_dataset=None):
    val_acc_ = []
    train_acc_ = []
    train_loss_ = []
    val_loss_ = []
    previsous_loss = 0
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        max_sequence_length = timesteps
        # Iterate over the batches of the dataset.
        tqdm_bar = tqdm(train_dataset)
        losses = []
        metrics = []

        for step, (x_batch_train, y_batch_train) in enumerate(tqdm_bar):
            if x_batch_train.shape[0] != batch_size:
                continue
                # x_batch_train = tf.concat([x_batch_train, tf.zeros((batch_size - x_batch_train.shape[0], max_sequence_length, 1629))], axis=0)

            loss_value = train_step(model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric,
                                    max_sequence_length)

            train_acc = train_acc_metric.result()
            losses.append(loss_value.numpy())
            metrics.append(train_acc)
            # Display metrics at the end of each 10 batch.
            tqdm_bar.set_postfix(train_loss=loss_value.numpy(), train_acc=float(train_acc))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        train_loss_.append(np.mean(losses))
        train_acc_.append(np.mean(metrics))
        print("Train acc: %.4f ; loss: %.4f" % (np.mean(metrics), np.mean(losses)))

        # Run a validation loop at the end of each epoch.
        if val_dataset is not None:
            tqdm_bar = tqdm(val_dataset)
            losses = []
            metrics = []
            for step, (x_batch_val, y_batch_val) in enumerate(tqdm_bar):
                if x_batch_val.shape[0] != batch_size:
                    continue

                val_loss = test_step(model, x_batch_val, y_batch_val, loss_fn, val_acc_metric, max_sequence_length)
                losses.append(val_loss.numpy())
                metrics.append(val_acc_metric.result())
                tqdm_bar.set_postfix(val_loss=val_loss.numpy(), val_acc=float(val_acc_metric.result()))
            val_acc_metric.reset_states()
            val_loss_.append(np.mean(losses))
            val_acc_.append(np.mean(metrics))
            previsous_loss = val_loss_[-1]
            print("Validation acc: %.4f ; loss: %.4f" % (np.mean(metrics), np.mean(losses)))
        print("Time taken: %.2fs" % (time.time() - start_time))

        # if epoch > 0 and val_loss_[-2] > previsous_loss:
        #     model.save_weights(f'checkpoint/lstm/model{epoch}.h5')
    return train_loss_, train_acc_, val_loss_, val_acc_


# %%

epochs = 55

metrics_ = custom_fit(model, epochs, train_dataset, val_dataset=val_dataset)

# %%
# plot the training loss and accuracy and validation loss and accuracy
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure(figsize=(20, 10))
plt.plot(N, metrics_[0], label="train_loss")
plt.plot(N, metrics_[2], label="val_loss")
plt.plot(N, metrics_[1], label="train_acc")
plt.plot(N, metrics_[3], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("plot_lstm64.png")

# %%
#tf.keras.saving.save_model(model, "../model_save")
# %%
