import keras
import numpy as np
import tensorflow as tf
import json
import tensorflow_addons as tfa

from custom_model.encoder_generator.utils import scaled_dot_product, scce_with_ls

# %%

# Global Variables and Constants

DATA_FOLDER = "/mnt/e/sign-lang-data-2"

LEFT_HAND_NAMES0 = []
INIT_ZEROS = tf.keras.initializers.constant(0.0)
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
GELU = tf.keras.activations.gelu

# Read Character to Ordinal Encoding Mapping
with open(f'{DATA_FOLDER}/character_to_prediction_index.json') as json_file:
    CHAR2ORD = json.load(json_file)

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


# %%

# Model and class Definition
class WeightDecayCallback(tf.keras.callbacks.Callback):
    """
        Callback to set weight decay of optimizer
    """
    def __init__(self, wd_ratio=0.05):
        super().__init__()
        self.step_counter = 0
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(
            f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')


class PreprocessLayer(tf.keras.layers.Layer):
    """
        Tensorflow layer to process data in TFLite
        Data needs to be processed in the model itself, so we can not use Python
    """

    def __init__(self):
        super(PreprocessLayer, self).__init__()
        self.normalisation_correction = tf.constant(
            # Add 0.50 to x coordinates of left hand (original right hand) and substract 0.50 of right hand (original left hand)
            [0.50 if 'x' in name else 0.00 for name in LEFT_HAND_NAMES0],
            dtype=tf.float32,
        )

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, 164], dtype=tf.float32),),
    )
    def call(self, data0, resize=True):
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

        # Pad Zeros
        N_FRAMES = len(data[0])
        if N_FRAMES < 128:
            data = tf.concat((
                data,
                tf.zeros([1, 128 - N_FRAMES, 164], dtype=tf.float32)
            ), axis=1)
        # Downsample
        data = tf.image.resize(
            data,
            [1, 128],
            method=tf.image.ResizeMethod.BILINEAR,
        )

        # Squeeze Batch Dimension
        data = tf.squeeze(data, axis=[0])

        return data


class LandmarkEmbedding(tf.keras.Model):
    """
        Embeds a landmark using fully connected layers. It is used to make the model invariant to the order of the landmarks.

    """
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False,
                                  kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False,
                                  kernel_initializer=INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def call(self, x, mask=None, training=None):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        )


# Creates embedding for each frame
class Embedding(tf.keras.Model):
    """
        Embeds the input data. It consists of a positional embedding and a landmark embedding.
        It is used to create representations for each frame.
    """
    def __init__(self, mean, std, n_frames, units):
        super(Embedding, self).__init__()
        self.mean = mean
        self.std = std

        self.n_frames = n_frames
        self.units = units

        self.supports_masking = True

    def build(self, input_shape):
        # Positional embedding for each frame index
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros([self.n_frames, self.units], dtype=tf.float32),
            trainable=True,
            name='embedding_positional_encoder',
        )
        # Embedding layer for Landmarks
        self.dominant_hand_embedding = LandmarkEmbedding(self.units, 'dominant_hand')

    def call(self, x, training=False, mask=None):
        # Normalize
        x = tf.where(
            tf.math.equal(x, 0.0),
            0.0,
            (x - self.mean) / self.std,
        )
        # Dominant Hand
        x = self.dominant_hand_embedding(x)
        # Add Positional Encoding
        x = x + self.positional_embedding

        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    """
        Multi Head Attention Layer
        The implementation of the Transformer is based on https://www.tensorflow.org/tutorials/text/transformer
    """
    def __init__(self, d_model, num_of_heads, dropout, d_out=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth // 2, use_bias=False) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth // 2, use_bias=False) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth // 2, use_bias=False) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model if d_out is None else d_out, use_bias=False)
        self.softmax = tf.keras.layers.Softmax()
        self.do = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, q, k, v, attention_mask=None, training=False):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](q)
            K = self.wk[i](k)
            V = self.wv[i](v)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        multi_head_attention = self.do(multi_head_attention, training=training)

        return multi_head_attention


# Encoder based on multiple transformer blocks
class Encoder(tf.keras.Model):
    """
        Encoder of the Transformer. It consists of multiple transformer blocks.
        It is used to create a representation of frames encoded in the embedding layer.
    """
    def __init__(self, num_blocks, units_encoder, num_head, mh_dropout_ratio, units_decoder, mlp_ratio,
                 mlp_dropout_ratio, n_frames, layer_nom_eps=1e-6):
        super(Encoder, self).__init__(name='encoder')
        self.mhas = None
        self.ln_1s = None
        self.ln_2s = None
        self.mlps = None
        self.supports_masking = True
        self.num_blocks = num_blocks
        self.units_encoder = units_encoder
        self.num_heads = num_head
        self.mha_dropout_ratio = mh_dropout_ratio
        self.mlp_ratio = mlp_ratio
        self.units_decoder = units_decoder
        self.layer_nom_eps = layer_nom_eps
        self.mlp_dropout_ratio = mlp_dropout_ratio
        self.n_frames = n_frames

    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_nom_eps))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.units_encoder, self.num_heads, self.mha_dropout_ratio))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_nom_eps))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(self.units_encoder * self.mlp_ratio, activation=GELU,
                                      kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False),
                tf.keras.layers.Dropout(self.mlp_dropout_ratio),
                tf.keras.layers.Dense(self.units_encoder, kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
            ]))
            # Optional Projection to Decoder Dimension
            if self.units_encoder != self.units_decoder:
                self.dense_out = tf.keras.layers.Dense(self.units_decoder, kernel_initializer=INIT_GLOROT_UNIFORM,
                                                       use_bias=False)
                self.apply_dense_out = True
            else:
                self.apply_dense_out = False

    def call(self, x, x_inp, training=False):
        # Attention mask to ignore missing frames
        attention_mask = tf.where(tf.math.reduce_sum(x_inp, axis=[2]) == 0.0, 0.0, 1.0)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = tf.repeat(attention_mask, repeats=self.n_frames, axis=1)
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x = ln_1(x + mha(x, x, x, attention_mask=attention_mask))
            x = ln_2(x + mlp(x))

        # Optional Projection to Decoder Dimension
        if self.apply_dense_out:
            x = self.dense_out(x)

        return x


# Decoder based on multiple transformer blocks
class Decoder(tf.keras.Model):
    """
        Decoder of the Transformer. It consists of multiple transformer blocks.
        Decoder_input is the output of the latent space encoder.
    """
    def __init__(self, num_blocks, units_decoder, num_head, mh_dropout_ratio, units_encoder, n_frames, mlp_ratio,
                 mlp_dropout_ratio, max_phrase_length, layer_nom_eps=1e-6):
        super(Decoder, self).__init__(name='decoder')
        self.num_blocks = num_blocks
        self.supports_masking = True
        self.units_decoder = units_decoder
        self.num_heads = num_head
        self.mha_dropout_ratio = mh_dropout_ratio
        self.units_encoder = units_encoder
        self.layer_nom_eps = layer_nom_eps
        self.n_frames = n_frames
        self.mlp_ratio = mlp_ratio
        self.mlp_dropout_ratio = mlp_dropout_ratio
        self.max_phrase_length = max_phrase_length

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros([self.n_frames, self.units_decoder], dtype=tf.float32),
            trainable=True,
            name='embedding_positional_encoder',
        )
        # Character Embedding
        self.char_emb = tf.keras.layers.Embedding(N_UNIQUE_CHARACTERS, self.units_decoder,
                                                  embeddings_initializer=INIT_ZEROS)
        # Positional Encoder MHA
        self.pos_emb_mha = MultiHeadAttention(self.units_decoder, self.num_heads, self.mha_dropout_ratio)
        self.pos_emb_ln = tf.keras.layers.LayerNormalization(epsilon=self.layer_nom_eps)
        # First Layer Normalisation
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_nom_eps))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.units_decoder, self.num_heads, self.mha_dropout_ratio))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_nom_eps))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(self.units_decoder * self.mlp_ratio, activation=GELU,
                                      kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False),
                tf.keras.layers.Dropout(self.mlp_dropout_ratio),
                tf.keras.layers.Dense(self.units_decoder, kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
            ]))

    def get_causal_attention_mask(self, B):
        i = tf.range(self.n_frames)[:, tf.newaxis]
        j = tf.range(self.n_frames)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, self.n_frames, self.n_frames))
        mult = tf.concat(
            [tf.expand_dims(B, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        mask = tf.tile(mask, mult)
        mask = tf.cast(mask, tf.float32)
        return mask

    def call(self, encoder_outputs, phrase, training=False):
        # Batch Size
        B = tf.shape(encoder_outputs)[0]
        # Cast to INT32
        phrase = tf.cast(phrase, tf.int32)
        # Prepend SOS Token
        phrase = tf.pad(phrase, [[0, 0], [1, 0]], constant_values=SOS_TOKEN, name='prepend_sos_token')
        # Pad With PAD Token
        phrase = tf.pad(phrase, [[0, 0], [0, self.n_frames - self.max_phrase_length - 1]], constant_values=PAD_TOKEN,
                        name='append_pad_token')
        # Causal Mask
        causal_mask = self.get_causal_attention_mask(B)
        # Positional Embedding
        x = self.positional_embedding + self.char_emb(phrase)
        # Causal Attention
        x = self.pos_emb_ln(x + self.pos_emb_mha(x, x, x, attention_mask=causal_mask))
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x = ln_1(x + mha(x, encoder_outputs, encoder_outputs, attention_mask=causal_mask))
            x = ln_2(x + mlp(x))
        # Slice 31 Characters
        x = tf.slice(x, [0, 0, 0], [-1, self.max_phrase_length, -1])

        return x


# Causal Attention to make decoder not attent to future characters which it needs to predict


def get_causal_attention_mask(B, n_frames):
    """
    Causal Attention to make decoder not attend to future characters which it needs to predict
    :param B:
    :param n_frames:
    :return:
    """
    i = tf.range(n_frames)[:, tf.newaxis]
    j = tf.range(n_frames)
    mask = tf.cast(i >= j, dtype=tf.int32)
    mask = tf.reshape(mask, (1, n_frames, n_frames))
    mult = tf.concat(
        [tf.expand_dims(B, -1), tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    mask = tf.tile(mask, mult)
    mask = tf.cast(mask, tf.float32)
    return mask


# TopK accuracy for multi dimensional output
class TopKAccuracy(tf.keras.metrics.Metric):
    """
    TopK accuracy for multi dimensional output
    """
    def __init__(self, k, **kwargs):
        super(TopKAccuracy, self).__init__(name=f'top{k}acc', **kwargs)
        self.top_k_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, N_UNIQUE_CHARACTERS])
        character_idxs = tf.where(y_true < N_UNIQUE_CHARACTERS0)
        y_true = tf.gather(y_true, character_idxs, axis=0)
        y_pred = tf.gather(y_pred, character_idxs, axis=0)
        self.top_k_acc.update_state(y_true, y_pred)

    def result(self):
        return self.top_k_acc.result()

    def reset_state(self):
        self.top_k_acc.reset_state()


def get_model(std, mean, params: dict):

    """
    This function returns the model for training with the given parameters
    :param std:
    :param mean:
    :param params:
    :return:
    """

    n_frames = params['n_frames']
    n_cols = params['n_cols']
    max_phrase_length = params['max_phrase_length']
    n_unique_characters = params['n_unique_characters']

    num_blocks_encoder = params['num_blocks_encoder']
    num_blocks_decoder = params['num_blocks_decoder']
    units_encoder = params['units_encoder']
    units_decoder = params['units_decoder']
    num_head = params['num_head']
    mlp_ratio = params['mlp_ratio']
    mh_dropout_ratio = params['mh_dropout_ratio']
    layer_nom_eps = params['layer_nom_eps']
    mlp_dropout_ratio = params['mlp_dropout_ratio']
    classifier_dropout_ratio = params['classifier_dropout_ratio']
    # model declaration
    frames_inp = tf.keras.layers.Input([n_frames, n_cols], dtype=tf.float32, name='frames')
    phrase_inp = tf.keras.layers.Input([max_phrase_length], dtype=tf.int32, name='phrase')
    # Frames
    x = frames_inp

    # Masking
    x = tf.keras.layers.Masking(mask_value=0.0, input_shape=(n_frames, n_cols))(x)

    # Embedding
    x = Embedding(std, mean, n_frames, units_encoder)(x)

    x = Encoder(
        num_blocks_encoder,
        units_encoder,
        num_head,
        mh_dropout_ratio,
        units_decoder,
        mlp_ratio,
        mlp_dropout_ratio,
        n_frames,
        layer_nom_eps
    )(x, frames_inp)

    # Decoder
    x = Decoder(
        num_blocks_decoder,
        units_decoder,
        num_head,
        mh_dropout_ratio,
        units_encoder,
        n_frames, mlp_ratio,
        mlp_dropout_ratio,
        max_phrase_length,
        layer_nom_eps
    )(x, phrase_inp)

    # Classifier
    x = tf.keras.Sequential([
        # Dropout
        tf.keras.layers.Dropout(classifier_dropout_ratio),
        # Output Neurons
        tf.keras.layers.Dense(n_unique_characters, activation=tf.keras.activations.linear,
                              kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
    ], name='classifier')(x)

    outputs = x

    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames_inp, phrase_inp], outputs=outputs)

    # model = AutoTranslatorModel(std=std, mean=mean)

    # Categorical Crossentropy Loss With Label Smoothing
    loss = scce_with_ls

    # Adam Optimizer
    optimizer = tfa.optimizers.RectifiedAdam(sma_threshold=4)
    optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)

    # TopK Metrics
    metrics = [
        TopKAccuracy(1),
        TopKAccuracy(5)
    ]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        loss_weights=loss_weights,
    )

    return model


class TFLiteModel(tf.Module):
    """
    This class is used to convert the model to tflite
    """
    def __init__(self, model):
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.preprocess_layer = PreprocessLayer()
        self.model = model

    @tf.function(jit_compile=True)
    def encoder(self, x, frames_inp):
        x = self.model.get_layer('embedding')(x)
        x = self.model.get_layer('encoder')(x, frames_inp)

        return x

    @tf.function(jit_compile=True)
    def decoder(self, x, phrase_inp):
        x = self.model.get_layer('decoder')(x, phrase_inp)
        x = self.model.get_layer('classifier')(x)

        return x

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 164], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        # Number Of Input Frames
        N_INPUT_FRAMES = tf.shape(inputs)[0]
        # Preprocess Data
        frames_inp = self.preprocess_layer(inputs)
        # Add Batch Dimension
        frames_inp = tf.expand_dims(frames_inp, axis=0)
        # Get Encoding
        encoding = self.encoder(frames_inp, frames_inp)
        # Make Prediction
        phrase = tf.fill([1, MAX_PHRASE_LENGTH], PAD_TOKEN)
        # Predict One Token At A Time
        stop = False
        for idx in tf.range(MAX_PHRASE_LENGTH):
            # Cast phrase to int8
            phrase = tf.cast(phrase, tf.int8)
            # If EOS token is predicted, stop predicting
            outputs = tf.cond(
                stop,
                lambda: tf.one_hot(tf.cast(phrase, tf.int32), N_UNIQUE_CHARACTERS),
                lambda: self.decoder(encoding, phrase)
            )
            # Add predicted token to input phrase
            phrase = tf.cast(phrase, tf.int32)
            # Replcae PAD token with predicted token up to idx
            phrase = tf.where(
                tf.range(MAX_PHRASE_LENGTH) < idx + 1,
                tf.argmax(outputs, axis=2, output_type=tf.int32),
                phrase,
            )
            # Predicted Token
            predicted_token = phrase[0, idx]
            # If EOS (End Of Sentence) token is predicted stop
            if not stop:
                stop = predicted_token == EOS_TOKEN

        # Squeeze outputs
        outputs = tf.squeeze(phrase, axis=0)
        outputs = tf.one_hot(outputs, N_UNIQUE_CHARACTERS)

        # Return a dictionary with the output tensor
        return {'outputs': outputs}
