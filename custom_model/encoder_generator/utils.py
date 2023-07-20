import numpy as np
import pandas as pd
import tensorflow as tf
import json
import math
from leven import levenshtein
from tqdm import tqdm
from matplotlib import pyplot as plt

DATA_FOLDER = "/mnt/e/sign-lang-data-2"
# Read Character to Ordinal Encoding Mapping
with open(f'{DATA_FOLDER}/character_to_prediction_index.json') as json_file:
    CHAR2ORD = json.load(json_file)
N_COLS0 = 164
N_COLS = 164
# Ordinal to Character Mapping
ORD2CHAR = {j: i for i, j in CHAR2ORD.items()}

N_UNIQUE_CHARACTERS0 = len(CHAR2ORD)
N_UNIQUE_CHARACTERS = len(CHAR2ORD) + 1 + 1 + 1

MAX_PHRASE_LENGTH = 31 + 1

PAD_TOKEN = len(CHAR2ORD)  # Padding
SOS_TOKEN = len(CHAR2ORD) + 1  # Start Of Sentence
EOS_TOKEN = len(CHAR2ORD) + 2  # End Of Sentence

# Create Initial Loss Weights All Set To 1
loss_weights = np.ones(N_UNIQUE_CHARACTERS, dtype=np.float32)
# Set Loss Weight Of Pad Token To 0
loss_weights[PAD_TOKEN] = 0
WD_RATIO = 0.05
WARMUP_METHOD = 'exp'


def scaled_dot_product(q, k, v, softmax, attention_mask):
    """
    based on: https://stackoverflow.com/questions/67342988/verifying-the-implementation-of-multihead-attention-in-transformer
    replaced softmax with softmax layer to support masked softmax
    :param q:
    :param k:
    :param v:
    :param softmax:
    :param attention_mask:
    :return:
    """
    # calculates Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # caculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt, mask=attention_mask)
    z = tf.matmul(softmax, v)
    # shape: (m,Tx,depth), same shape as q,k,v
    return z


def get_causal_attention_mask(B, N_TARGET_FRAMES=128):
    """
    Causal Attention to make decoder not attend to future characters which it needs to predict
    :param B:
    :param N_TARGET_FRAMES:
    :return:
    """
    i = tf.range(N_TARGET_FRAMES)[:, tf.newaxis]
    j = tf.range(N_TARGET_FRAMES)
    mask = tf.cast(i >= j, dtype=tf.int32)
    mask = tf.reshape(mask, (1, N_TARGET_FRAMES, N_TARGET_FRAMES))
    mult = tf.concat(
        [tf.expand_dims(B, -1), tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    mask = tf.tile(mask, mult)
    mask = tf.cast(mask, tf.float32)
    return mask


def scce_with_ls(y_true, y_pred):
    """
    Sparse Categorical Crossentropy with Label Smoothing
    :param y_true:
    :param y_pred:
    :return:
    """
    # Filter Pad Tokens
    idxs = tf.where(y_true != PAD_TOKEN)
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, N_UNIQUE_CHARACTERS, axis=1)
    # Categorical Crossentropy with native label smoothing support
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25, from_logits=True)
    loss = tf.math.reduce_mean(loss)
    return loss


def plot_lr_schedule(lr_schedule, epochs):
    """
    Plot Learning Rate Schedule
    :param lr_schedule:
    :param epochs:
    :return:
    """
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels)  # set tick step to 1 and let x axis start at 1

    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])

    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)

    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black')
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)

    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()


def lrfn(current_step, num_warmup_steps, lr_max, num_training_steps, num_cycles=0.50):
    """
    Learning Rate Schedule
    :param current_step:
    :param num_warmup_steps:
    :param lr_max:
    :param num_training_steps:
    :param num_cycles:
    :return:
    """
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max


# Output Predictions to string
def outputs2phrase(outputs):
    """
    Convert Model Outputs to String
    :param outputs:
    :return:
    """
    if outputs.ndim == 2:
        outputs = np.argmax(outputs, axis=1)

    return ''.join([ORD2CHAR.get(s, '') for s in outputs])


def plot_history_metric(history, metric, model_name, f_best=np.argmax, ylim=None, yscale=None, yticks=None):
    """
    Plot History Metric
    plot the best point model with the metric
    :param history:
    :param metric:
    :param model_name:
    :param f_best:
    :param ylim:
    :param yscale:
    :param yticks:
    :return:
    """

    # Only plot when training

    plt.figure(figsize=(20, 10))

    values = history.history[metric]
    N_EPOCHS = len(values)
    val = 'val' in ''.join(history.history.keys())
    # Epoch Ticks
    if N_EPOCHS <= 20:
        x = np.arange(1, N_EPOCHS + 1)
    else:
        x = [1, 5] + [10 + 5 * idx for idx in range((N_EPOCHS - 10) // 5 + 1)]

    x_ticks = np.arange(1, N_EPOCHS + 1)

    # Validation
    if val:
        val_values = history.history[f'val_{metric}']
        val_argmin = f_best(val_values)
        plt.plot(x_ticks, val_values, label=f'val')

    # summarize history for accuracy
    plt.plot(x_ticks, values, label=f'train')
    argmin = f_best(values)
    plt.scatter(argmin + 1, values[argmin], color='red', s=75, marker='o', label=f'train_best')
    if val:
        plt.scatter(val_argmin + 1, val_values[val_argmin], color='purple', s=75, marker='o', label=f'val_best')

    plt.title(f'Model {metric}', fontsize=24, pad=10)
    plt.ylabel(metric, fontsize=20, labelpad=10)

    if ylim:
        plt.ylim(ylim)

    if yscale is not None:
        plt.yscale(yscale)

    if yticks is not None:
        plt.yticks(yticks, fontsize=16)

    plt.xlabel('epoch', fontsize=20, labelpad=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.xticks(x, fontsize=16)  # set tick step to 1 and let x axis start at 1
    plt.yticks(fontsize=16)

    plt.legend(prop={'size': 10})
    plt.grid()
    plt.savefig(f'./{model_name}_{metric}.png')
    plt.show()


#
def verify_no_nan_predictions(model, dataset, steps=100):
    """
    Verify No NaN Predictions
    :param model:
    :param dataset:
    :param steps:
    :return:
    """
    y_pred = model.predict(
        dataset,
        steps=steps,
    )

    print(f'# NaN Values In Predictions: {np.isnan(y_pred).sum()}')

    plt.figure(figsize=(15, 8))
    plt.title(f'Logit Predictions Initialized Model')
    pd.Series(y_pred.flatten()).plot(kind='hist', bins=128)
    plt.xlabel('Logits')
    plt.grid()
    plt.show()


def verify_correct_training_flag(model, X_batch_small):
    """
    Verify Correct Training Flag
    :param model:
    :param X_batch_small:
    :return:
    """
    # Verify static output for inference
    pred = model(X_batch_small, training=False)
    for _ in tqdm(range(10)):
        assert tf.reduce_min(tf.cast(pred == model(X_batch_small, training=False), tf.int8)) == 1

    # Verify at least 99% varying output due to dropout during training
    for _ in tqdm(range(10)):
        assert tf.reduce_mean(tf.cast(pred != model(X_batch_small, training=True), tf.float32)) > 0.99


@tf.function()
def predict_phrase(frames, model):
    """
    Predict Phrase
    :param frames:
    :param model:
    :return:
    """
    # Add Batch Dimension
    frames = tf.expand_dims(frames, axis=0)
    # Start Phrase
    phrase = tf.fill([1, MAX_PHRASE_LENGTH], PAD_TOKEN)

    for idx in tf.range(MAX_PHRASE_LENGTH):
        # Cast phrase to int8
        phrase = tf.cast(phrase, tf.int8)
        # Predict Next Token
        outputs = model({
            'frames': frames,
            'phrase': phrase,
        })

        # Add predicted token to input phrase
        phrase = tf.cast(phrase, tf.int32)
        phrase = tf.where(
            tf.range(MAX_PHRASE_LENGTH) < idx + 1,
            tf.argmax(outputs, axis=2, output_type=tf.int32),
            phrase,
        )

    # Squeeze outputs
    outputs = tf.squeeze(phrase, axis=0)
    outputs = tf.one_hot(outputs, N_UNIQUE_CHARACTERS)

    # Return a dictionary with the output tensor
    return outputs


def get_ld_train(x_train, y_train, model):
    """
    Get Levenstein Distance Train
    :param x_train:
    :param y_train:
    :param model:
    :return:
    """
    N = 1000
    LD_TRAIN = []
    for idx, (frames, phrase_true) in enumerate(zip(tqdm(x_train, total=N), y_train)):
        # Predict Phrase and Convert to String
        phrase_pred = predict_phrase(frames, model).numpy()
        phrase_pred = outputs2phrase(phrase_pred)
        # True Phrase Ordinal to String
        phrase_true = outputs2phrase(phrase_true)
        # Add Levenstein Distance
        LD_TRAIN.append({
            'phrase_true': phrase_true,
            'phrase_pred': phrase_pred,
            'levenshtein_distance': levenshtein(phrase_pred, phrase_true),
        })
        # Take subset in interactive mode
        if idx == N:
            break

    # Convert to DataFrame
    LD_TRAIN_DF = pd.DataFrame(LD_TRAIN)

    return LD_TRAIN_DF


def get_ld_val(x_val, y_val, model):
    """
    Get Levenstein Distance Val
    :param x_val:
    :param y_val:
    :param model:
    :return:
    """
    N = 1000
    LD_VAL = []
    for idx, (frames, phrase_true) in enumerate(zip(tqdm(x_val, total=N), y_val)):
        # Predict Phrase and Convert to String
        phrase_pred = predict_phrase(frames, model).numpy()
        phrase_pred = outputs2phrase(phrase_pred)
        # True Phrase Ordinal to String
        phrase_true = outputs2phrase(phrase_true)
        # Add Levenstein Distance
        LD_VAL.append({
            'phrase_true': phrase_true,
            'phrase_pred': phrase_pred,
            'levenshtein_distance': levenshtein(phrase_pred, phrase_true),
        })
        # Take subset in interactive mode
        if idx == N:
            break

    # Convert to DataFrame
    LD_VAL_DF = pd.DataFrame(LD_VAL)

    return LD_VAL_DF


def get_file_path(path):
    return f'{DATA_FOLDER}/{path}'


# Train Dataset Iterator
def get_train_dataset(X, y, batch_size):
    """
    Get Train Dataset Iterator
    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    sample_idxs = np.arange(len(X))
    while True:
        # Get random indices
        random_sample_idxs = np.random.choice(sample_idxs, batch_size)

        inputs = {
            'frames': X[random_sample_idxs],
            'phrase': y[random_sample_idxs],
        }
        outputs = y[random_sample_idxs]

        yield inputs, outputs


# Validation Dataset Iterator
def get_val_dataset(X, y, batch_size):
    """
    Get Validation Dataset Iterator
    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    offsets = np.arange(0, len(X), batch_size)
    while True:
        # Iterate over whole validation set
        for offset in offsets:
            inputs = {
                'frames': X[offset:offset + batch_size],
                'phrase': y[offset:offset + batch_size],
            }
            outputs = y[offset:offset + batch_size]

            yield inputs, outputs


def get_idxs(df, words_pos, words_neg=[], ret_names=True, idxs_pos=None):
    """
    Get Idxs from Dataframe
    :param df:
    :param words_pos:
    :param words_neg:
    :param ret_names:
    :param idxs_pos:
    :return:
    """
    idxs = []
    names = []
    for w in words_pos:
        for col_idx, col in enumerate(df.columns):
            # Exclude Non Landmark Columns
            if col in ['frame']:
                continue

            col_idx = int(col.split('_')[-1])
            # Check if column name contains all words
            if (w in col) and (idxs_pos is None or col_idx in idxs_pos) and all([w not in col for w in words_neg]):
                idxs.append(col_idx)
                names.append(col)
    # Convert to Numpy arrays
    idxs = np.array(idxs)
    names = np.array(names)
    # Returns either both column indices and names
    if ret_names:
        return idxs, names
    # Or only columns indices
    else:
        return idxs


class PreprocessLayerNonNaN(tf.keras.layers.Layer):
    """
    Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
    """

    def __init__(self):
        super(PreprocessLayerNonNaN, self).__init__()

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, N_COLS0], dtype=tf.float32),),
    )
    def call(self, data0):
        # Fill NaN Values With 0
        data = tf.where(tf.math.is_nan(data0), 0.0, data0)

        # Hacky
        data = data[None]

        # Empty Hand Frame Filtering
        hands = tf.slice(data, [0, 0, 0], [-1, -1, 84])
        hands = tf.abs(hands)
        mask = tf.reduce_sum(hands, axis=2)
        mask = tf.not_equal(mask, 0)
        data = data[mask][None]
        data = tf.squeeze(data, axis=[0])

        return data
