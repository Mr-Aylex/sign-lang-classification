import tensorflow as tf
import keras
from keras.layers import GRU, Masking, Dense
from keras import Input
import numpy as np
import time
from tqdm import tqdm


class CustomModel(keras.Model):

    def __init__(self, batch_size, timesteps, features, nb_classes):
        super().__init__()
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.features = features

        self.masking = Masking(mask_value=0., batch_input_shape=(batch_size, timesteps, features))
        self.lstm = GRU(64, stateful=True, input_dim=(timesteps, features))
        self.output_ = Dense(units=nb_classes, activation='softmax')

    def call(self, inputs, training=False, mask=None):

        x = self.masking(inputs)
        x = self.lstm(x)
        x = self.output_(x)
        return x

    @tf.function
    def train_step2(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        max_sequence_length = self.timesteps
        x, y = data
        # structure of x = (batch_size, timesteps, features)

        metrics = {}
        # for batch in range(x.shape[0]):
        batch = 0
        # On decoupe la sequence en plusieurs sequences de taille max_sequence_length
        for seq in range(x.shape[1] // max_sequence_length):

            with tf.GradientTape() as tape:
                y_pred = self(x[0:1, seq:seq + max_sequence_length], training=True)  # Forward pass
                # Compute the loss value

                loss = self.compiled_loss(y[0:1, 0], y_pred,
                                          regularization_losses=self.losses)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y[0:1, 0], y_pred)
            # On recupere les metrics
            for m in self.metrics:
                if m.name not in metrics:
                    metrics[m.name] = []
                metrics[m.name].append(m.result())
        self.reset_states()
        return {m.name: tf.reduce_mean(metrics[m.name]) for m in self.metrics}

    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        max_sequence_length = self.timesteps
        x, y = data
        # structure of x = (batch_size, timesteps, features)
        metrics = {}

        for seq in range(self.timesteps // max_sequence_length):
            with tf.GradientTape() as tape:
                logits = self(x[:, seq:seq + max_sequence_length], training=True)
                loss_value = self.compiled_loss(y[:, :], logits)

            grads = tape.gradient(loss_value, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reset_states()
        # Update training metric.
        self.compiled_metrics.update_state(y[:, :], logits)
        return {m.name: tf.reduce_mean(metrics[m.name]) for m in self.compiled_metrics}

    # custom evaluation step

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        max_sequence_length = self.timesteps
        # Compute predictions
        metrics = {}
        batch = 0
        for seq in range(self.timesteps // max_sequence_length):
            logits = self(x[:, seq:seq + max_sequence_length], training=True)
            # loss_value = self.compiled_loss(y[:, :], logits)
        self.reset_states()
        # Update training metric.
        self.metrics.update_state(y[:, :], logits)
        return {m.name: tf.reduce_mean(metrics[m.name]) for m in self.compiled_metrics}

    def custom_fit(self, train_dataset, val_dataset, epochs):
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            max_sequence_length = self.timesteps
            # Iterate over the batches of the dataset.
            tqdm_bar = tqdm(train_dataset)
            for step, (x_batch_train, y_batch_train) in enumerate(tqdm_bar):
                if x_batch_train.shape[0] != self.batch_size:
                    continue
                    # x_batch_train = tf.concat([x_batch_train, tf.zeros((batch_size - x_batch_train.shape[0], max_sequence_length, 1629))], axis=0)

                loss_value = self.train_step(x_batch_train, y_batch_train)

                train_acc = self.compiled_metrics.result()
                # Display metrics at the end of each 10 batch.
                if step % 10 == 0:
                    tqdm_bar.set_postfix(train_loss=loss_value.numpy(), train_acc=float(train_acc))

            # Reset training metrics at the end of each epoch
            self.compiled_metrics.reset_states()

            # Run a validation loop at the end of each epoch.
            if val_dataset is not None:
                tqdm_bar = tqdm(val_dataset)
                for step, (x_batch_val, y_batch_val) in enumerate(tqdm_bar):
                    if x_batch_val.shape[0] != self.batch_size:
                        continue

                    val_loss = self.test_step(x_batch_val, y_batch_val)
                    if step % 10 == 0:
                        tqdm_bar.set_postfix(val_loss=val_loss.numpy(), val_acc=float(self.compiled_metrics.result()))
                self.compiled_metrics.reset_states()
