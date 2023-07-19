# from pathlib import Path
#
# import tensorflow as tf
# import keras
# from keras import utils, optimizers, losses, metrics
# import numpy as np
# from keras import callbacks
# from keras.layers import TimeDistributed
# from keras import layers
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import keras_nlp
# import os
# import json
# import pandas as pd
# from custum_translate_model.first import Translator1, Translator2, Translator3, Translator4, Translator5, VarTranslator, \
#     VarTranslator2
#
# # %%
#
# # set seed
# utils.set_random_seed(42)
#
# # %%
# MAX_NB_FRAMES = 900
# DATA_FOLDER = "/mnt/e/sign-lang-data-2"
# train = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv"))
#
# with open("monBytePairTokenizer.json", "r") as f:
#     js_file = json.load(f)
# # %%
# vocab = js_file["model"]["vocab"]
# merge = js_file["model"]["merges"]
# # %%
# tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocab, merge)
#
#
# # %%
# def load_npy(npy_path, phrase):
#     max_nb_frames = MAX_NB_FRAMES
#
#     filename = npy_path.numpy().decode("utf-8")
#     try:
#         inputs = np.load(filename)
#     except:
#         print(filename)
#         raise Exception("error")
#     # fill nan with 0
#     inputs = np.nan_to_num(inputs)
#     if inputs.shape[0] > max_nb_frames:
#         inputs = inputs[:max_nb_frames, :, :]
#     else:
#         inputs = np.concatenate([inputs, np.zeros((max_nb_frames - inputs.shape[0], 543, 3))], axis=0)
#
#     return inputs, phrase.numpy().decode("utf-8")
#
#
# def add_mask(inputs, label):
#     label = label.numpy()
#     shape = label.shape
#     new_label = np.full(20, 4)
#     new_label[:shape[1]] = label[0]
#     return inputs, new_label
#
#
# def load_dataset(dataset):
#     # load only the npy
#     dataset = dataset.map(
#         lambda npy_path, phrase: tf.py_function(func=load_npy, inp=[npy_path, phrase], Tout=(tf.float32, tf.string)),
#         num_parallel_calls=tf.data.AUTOTUNE)
#
#     dataset = dataset.map(
#         lambda inputs, label: (inputs, tokenizer.tokenize(label)),
#         num_parallel_calls=tf.data.AUTOTUNE)
#
#     dataset = dataset.map(
#         lambda inputs, label: tf.py_function(func=add_mask, inp=[inputs, label], Tout=(tf.float32, tf.int32)),
#         num_parallel_calls=tf.data.AUTOTUNE)
#
#     # reshape the input
#     dataset = dataset.map(
#         lambda inputs, label: (tf.reshape(inputs, (900, 543, 3)), label),
#         num_parallel_calls=tf.data.AUTOTUNE)
#
#     return dataset
#
#
# # %%
#
# train["npy_path"] = train[["sequence_id", "path"]].apply(
#     lambda x: os.path.join(DATA_FOLDER, "npy_train_landmarks", x["path"].replace(".parquet", "").replace("/", "_"),
#                            str(x["sequence_id"]) + ".npy"),
#     axis=1
# )
#
# # %%
#
# # split the dataset into train and validation set (80/20)
#
# X = train["npy_path"]
# Y = train["phrase"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = load_dataset(train_dataset)
#
# validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# validation_dataset = load_dataset(validation_dataset)
#
# # %%
#
# batch_size = 8
#
# train_dataset = train_dataset.batch(batch_size)
# train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
#
# validation_dataset = validation_dataset.batch(batch_size)
# validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
#
# # %%
#
#
# #translator1 = Translator1(64, vocab, merge, 64)
# #translator2 = Translator2(64, vocab, merge, 64)
# #translator3 = Translator3(64, vocab, merge, 64)
# #translator4 = Translator4(64, vocab, merge, 64)
# #translator5 = Translator5(64, vocab, merge, 64)
#
# var_translator = VarTranslator(20, vocab, merge, 20)
#
# # %%
# # do call
# out = var_translator(np.zeros((batch_size, 900, 543, 3), dtype=np.float32))
#
# print(var_translator.summary())
# print(out.shape)
#
#
# for model in [var_translator]:
#
#     out = model(np.zeros((batch_size, 900, 543, 3)))
#     Path(f"checkpoint/{model.name}").mkdir(parents=True, exist_ok=True)
#     checkpoint = callbacks.ModelCheckpoint(filepath=f'checkpoint/{model.name}/' + 'model_{epoch}.h5',
#                                            monitor='val_loss', save_best_only=True, save_weights_only=True)
#     nan = callbacks.TerminateOnNaN()
#     csv_logger = callbacks.CSVLogger(f'training_{model.name}.csv')
#
#
#     def scheduler(epoch, lr):
#         return lr * tf.math.exp(-0.2)
#
#
#     exp_scheduler = callbacks.LearningRateScheduler(scheduler)
#
#     print(model.summary())
#     model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
#                        loss=losses.SparseCategoricalCrossentropy(),
#                        metrics=[metrics.SparseCategoricalAccuracy()])
#
#     model.fit(train_dataset, epochs=10,
#               validation_data=validation_dataset,
#               callbacks=[checkpoint, nan, csv_logger, exp_scheduler]
#               )
