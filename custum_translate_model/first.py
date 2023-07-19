import tensorflow as tf
import keras
from keras import utils, optimizers, losses, metrics
import numpy as np
from keras import layers
from keras.layers import TimeDistributed
from tqdm import tqdm
import keras_nlp
import os
import json


# %%


class VarTranslator(keras.Model):
    def __init__(self, max_sequence_length, vocabulaire, merge, latent_dim, *args, **kwargs):
        super().__init__(name="VarAutoEncoder", *args, **kwargs)

        self.latent_dim = latent_dim
        self.tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocabulaire, merge)
        # encoder
        encoder = keras.Sequential([
            layers.Conv1D(64, 9, activation='relu'),
            layers.MaxPool1D(2),
            layers.Conv1D(32, 6, activation='relu'),
            layers.MaxPool1D(2),
            layers.Conv1D(16, 3, activation='relu'),
            layers.MaxPool1D(2),
            layers.Conv1D(8, 3, activation='relu'),
            layers.MaxPool1D(2),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.mask = layers.Masking(mask_value=0.0)
        self.encoder = TimeDistributed(encoder)
        self.lstm = layers.LSTM(max_sequence_length)
        self.reshape = layers.Reshape((max_sequence_length, 1))

        self.dense = layers.Dense(self.tokenizer.vocabulary_size(), activation='linear')

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # encoder
        x = self.mask(inputs)
        x = self.encoder(x)

        # x = self.flatten(x)
        x = self.lstm(x)
        x = self.reshape(x)
        x = self.dense(x)
        return x


class VarTranslator2(keras.Model):
    def __init__(self, max_sequence_length, vocabulaire, merge, latent_dim, *args, **kwargs):
        super().__init__(name="VarAutoEncoder2", *args, **kwargs)

        self.latent_dim = latent_dim
        self.tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocabulaire, merge)
        # encoder
        self.conv = TimeDistributed(layers.Conv1D(64, 9, activation='relu'))
        self.pool = TimeDistributed(layers.MaxPool1D(2))
        self.conv2 = TimeDistributed(layers.Conv1D(32, 6, activation='relu'))
        self.pool2 = TimeDistributed(layers.MaxPool1D(2))
        self.conv3 = TimeDistributed(layers.Conv1D(16, 3, activation='relu'))
        self.pool3 = TimeDistributed(layers.MaxPool1D(2))
        self.conv4 = TimeDistributed(layers.Conv1D(8, 3, activation='relu'))
        self.pool4 = TimeDistributed(layers.MaxPool1D(2))

        # decoder
        self.repeat = layers.RepeatVector(max_sequence_length)
        self.gru2 = layers.GRU(8, return_sequences=True)
        self.dense = layers.Dense(self.tokenizer.vocabulary_size(), activation='linear')

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # encoder
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)

        print(x.shape)
        # decoder
        x = self.repeat(x)
        x = self.gru2(x)
        x = self.dense(x)

        return x


class Translator1(keras.Model):

    def __init__(self, max_sequence_length, vocabulaire, merge, latent_dim, *args, **kwargs):
        super().__init__(name="AutoEncoder", *args, **kwargs)
        # encoder part
        self.latent_dim = latent_dim
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(64, 3, activation='relu')
        self.pool2 = layers.MaxPool2D(2)
        self.conv3 = layers.Conv2D(128, 3, activation='relu')
        self.pool3 = layers.MaxPool2D(2)
        self.conv4 = layers.Conv2D(256, 3, activation='relu')
        self.pool4 = layers.MaxPool2D(2)
        self.conv5 = layers.Conv2D(1, 3, activation='relu', padding='same')
        self.reshape = layers.Reshape((64, 27))

        # decoder part
        self.max_sequence_length = max_sequence_length
        self.vocabulaire = vocabulaire
        self.tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocabulaire, merge)

        self.transformer_decoder = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=64, num_heads=2
        )
        # self.fnet = keras_nlp.layers.FNetEncoder(intermediate_dim=32)
        self.dense = layers.Dense(self.tokenizer.vocabulary_size(), activation='linear')

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # encoder part
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.reshape(x)
        # decoder part
        x = self.transformer_decoder(x)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_sequence_length": self.max_sequence_length,
            "vocabulaire": self.vocabulaire,
            "merge": self.merge,
            "latent_dim": self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %%

class Translator2(keras.Model):
    def __init__(self, max_sequence_length, vocabulaire, merge, latent_dim, dropout_rate=0.4, *args, **kwargs):
        super().__init__(name="AutoEncoder2", *args, **kwargs)
        # encoder part
        self.latent_dim = latent_dim
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.drop1 = layers.SpatialDropout2D(dropout_rate)
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(64, 3, activation='relu')
        self.drop2 = layers.SpatialDropout2D(dropout_rate)
        self.pool2 = layers.MaxPool2D(2)
        self.conv3 = layers.Conv2D(128, 3, activation='relu')
        self.drop3 = layers.SpatialDropout2D(dropout_rate)
        self.pool3 = layers.MaxPool2D(2)
        self.conv4 = layers.Conv2D(256, 3, activation='relu')
        self.drop4 = layers.SpatialDropout2D(dropout_rate)
        self.pool4 = layers.MaxPool2D(2)
        self.conv5 = layers.Conv2D(1, 3, activation='relu', padding='same')
        self.reshape = layers.Reshape((64, 27))

        # decoder part
        self.max_sequence_length = max_sequence_length
        self.vocabulaire = vocabulaire
        self.tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocabulaire, merge)

        self.transformer_decoder = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=64, num_heads=2
        )
        # self.fnet = keras_nlp.layers.FNetEncoder(intermediate_dim=32)
        self.dense = layers.Dense(self.tokenizer.vocabulary_size(), activation='linear')

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # encoder part
        x = self.conv1(inputs)
        x = self.drop1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.drop3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.drop4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.reshape(x)
        # decoder part
        x = self.transformer_decoder(x)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_sequence_length": self.max_sequence_length,
            "vocabulaire": self.vocabulaire,
            "merge": self.merge,
            "latent_dim": self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Translator3(keras.Model):
    def __init__(self, max_sequence_length, vocabulaire, merge, latent_dim, dropout_rate=0.4, *args, **kwargs):
        super().__init__(name="AutoEncoder3", *args, **kwargs)
        # encoder part
        self.latent_dim = latent_dim
        self.recurent = layers.ConvLSTM1D(64, 3, activation='relu', padding='same')
        self.reshape = layers.Reshape((543, 64, 1))
        self.recurent2 = layers.ConvLSTM1D(32, 3, activation='relu', padding='same')

        # decoder part
        self.max_sequence_length = max_sequence_length
        self.vocabulaire = vocabulaire
        self.tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocabulaire, merge)

        self.transformer_decoder = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=64, num_heads=2
        )
        # self.fnet = keras_nlp.layers.FNetEncoder(intermediate_dim=32)
        self.dense = layers.Dense(self.tokenizer.vocabulary_size(), activation='linear')

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # encoder part
        x = self.recurent(inputs)
        x = self.reshape(x)
        x = self.recurent2(x)
        # decoder part
        x = self.transformer_decoder(x)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_sequence_length": self.max_sequence_length,
            "vocabulaire": self.vocabulaire,
            "merge": self.merge,
            "latent_dim": self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Translator4(keras.Model):

    def __init__(self, max_sequence_length, vocabulaire, merge, latent_dim, *args, **kwargs):
        super().__init__(name="AutoEncoder4", *args, **kwargs)
        # encoder part
        self.latent_dim = latent_dim
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.conv3 = layers.Conv2D(64, 3, activation='relu')
        self.pool3 = layers.MaxPool2D(4)
        self.conv4 = layers.Conv2D(128, 3, activation='relu')
        self.pool4 = layers.MaxPool2D(2)
        self.conv5 = layers.Conv2D(1, 3, activation='relu', padding='valid')
        self.reshape = layers.Reshape((64, 109))

        # decoder part
        self.max_sequence_length = max_sequence_length
        self.vocabulaire = vocabulaire
        self.tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocabulaire, merge)

        self.transformer_decoder = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=64, num_heads=2
        )
        # self.fnet = keras_nlp.layers.FNetEncoder(intermediate_dim=32)
        self.dense = layers.Dense(self.tokenizer.vocabulary_size(), activation='linear')

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # encoder part
        x = self.conv1(inputs)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.reshape(x)
        # decoder part
        x = self.transformer_decoder(x)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_sequence_length": self.max_sequence_length,
            "vocabulaire": self.vocabulaire,
            "merge": self.merge,
            "latent_dim": self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Translator5(keras.Model):

    def __init__(self, max_sequence_length, vocabulaire, merge, latent_dim, dropout_rate=0.6, *args, **kwargs):
        super().__init__(name="AutoEncoder5", *args, **kwargs)
        # encoder part
        self.latent_dim = latent_dim
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.drop1 = layers.SpatialDropout2D(dropout_rate)
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(64, 3, activation='relu')
        self.drop2 = layers.SpatialDropout2D(dropout_rate)
        self.pool2 = layers.MaxPool2D(2)
        self.conv3 = layers.Conv2D(128, 3, activation='relu')
        self.drop3 = layers.SpatialDropout2D(dropout_rate)
        self.pool3 = layers.MaxPool2D(2)
        self.conv4 = layers.Conv2D(256, 3, activation='relu')
        self.drop4 = layers.SpatialDropout2D(dropout_rate)
        self.pool4 = layers.MaxPool2D(2)
        self.conv5 = layers.Conv2D(1, 3, activation='relu', padding='same')
        self.reshape = layers.Reshape((64, 27))

        # decoder part
        self.max_sequence_length = max_sequence_length
        self.vocabulaire = vocabulaire
        self.tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocabulaire, merge)

        self.transformer_decoder = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=64, num_heads=3
        )
        # self.fnet = keras_nlp.layers.FNetEncoder(intermediate_dim=32)
        self.dense = layers.Dense(self.tokenizer.vocabulary_size(), activation='linear')

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # encoder part
        x = self.conv1(inputs)
        x = self.drop1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.drop3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.drop4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.reshape(x)
        # decoder part
        x = self.transformer_decoder(x)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_sequence_length": self.max_sequence_length,
            "vocabulaire": self.vocabulaire,
            "merge": self.merge,
            "latent_dim": self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
