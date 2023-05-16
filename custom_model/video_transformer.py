import tensorflow as tf
from keras import layers
from keras import mixed_precision
from keras.optimizers.adamw import AdamW
import keras

import random
from matplotlib import pyplot as plt

# Set seed for reproducibility.
keras.utils.set_random_seed(42)

AUTO = tf.data.AUTOTUNE

config = {
    "mixed_precision": True,
    "dataset": "cifar10",
    "train_slice": 40_000,
    "batch_size": 2048,
    "buffer_size": 2048 * 2,
    "input_shape": [32, 32, 3],
    "image_size": 48,
    "num_classes": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 30,
    "patch_size": 4,
    "embed_dim": 64,
    "chunk_size": 8,
    "r": 2,
    "num_layers": 4,
    "ffn_drop": 0.2,
    "attn_drop": 0.2,
    "num_heads": 1,
}

if config["mixed_precision"]:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_val, y_val) = (
    (x_train[: config["train_slice"]], y_train[: config["train_slice"]]),
    (x_train[config["train_slice"] :], y_train[config["train_slice"] :]),
)



# Build the `train` augmentation pipeline.
train_augmentation = keras.Sequential(
    [
        layers.Rescaling(1 / 255.0, dtype="float32"),
        layers.Resizing(
            config["input_shape"][0] + 20,
            config["input_shape"][0] + 20,
            dtype="float32",
        ),
        layers.RandomCrop(config["image_size"], config["image_size"], dtype="float32"),
        layers.RandomFlip("horizontal", dtype="float32"),
    ],
    name="train_data_augmentation",
)

# Build the `val` and `test` data pipeline.
test_augmentation = keras.Sequential(
    [
        layers.Rescaling(1 / 255.0, dtype="float32"),
        layers.Resizing(config["image_size"], config["image_size"], dtype="float32"),
    ],
    name="test_data_augmentation",
)

# We define functions in place of simple lambda functions to run through the
# [`keras.Sequential`](/api/models/sequential#sequential-class)in order to solve this warning:
# (https://github.com/tensorflow/tensorflow/issues/56089)


def train_map_fn(image, label):
    return train_augmentation(image), label


def test_map_fn(image, label):
    return test_augmentation(image), label



train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds.map(train_map_fn, num_parallel_calls=AUTO)
    .shuffle(config["buffer_size"])
    .batch(config["batch_size"], num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = (
    val_ds.map(test_map_fn, num_parallel_calls=AUTO)
    .batch(config["batch_size"], num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = (
    test_ds.map(test_map_fn, num_parallel_calls=AUTO)
    .batch(config["batch_size"], num_parallel_calls=AUTO)
    .prefetch(AUTO)
)


class PatchEmbedding(layers.Layer):
    """Image to Patch Embedding.
    Args:
        image_size (`Tuple[int]`): Size of the input image.
        patch_size (`Tuple[int]`): Size of the patch.
        embed_dim (`int`): Dimension of the embedding.
        chunk_size (`int`): Number of patches to be chunked.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        chunk_size,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Compute the patch resolution.
        patch_resolution = [
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        ]

        # Store the parameters.
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]

        # Define the positions of the patches.
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

        # Create the layers.
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            name="projection",
        )
        self.flatten = layers.Reshape(
            target_shape=(-1, embed_dim),
            name="flatten",
        )
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=embed_dim,
            name="position_embedding",
        )
        self.layernorm = keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="layernorm",
        )
        self.chunking_layer = layers.Reshape(
            target_shape=(self.num_patches // chunk_size, chunk_size, embed_dim),
            name="chunking_layer",
        )

    def call(self, inputs):
        # Project the inputs to the embedding dimension.
        x = self.projection(inputs)

        # Flatten the pathces and add position embedding.
        x = self.flatten(x)
        x = x + self.position_embedding(self.positions)

        # Normalize the embeddings.
        x = self.layernorm(x)

        # Chunk the tokens.
        x = self.chunking_layer(x)

        return x

class FeedForwardNetwork(layers.Layer):
    """Feed Forward Network.
    Args:
        dims (`int`): Number of units in FFN.
        dropout (`float`): Dropout probability for FFN.
    """

    def __init__(self, dims, dropout, **kwargs):
        super().__init__(**kwargs)

        # Create the layers.
        self.ffn = keras.Sequential(
            [
                layers.Dense(units=4 * dims, activation=tf.nn.gelu),
                layers.Dense(units=dims),
                layers.Dropout(rate=dropout),
            ],
            name="ffn",
        )
        self.layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="layernorm",
        )

    def call(self, inputs):
        # Apply the FFN.
        x = self.layernorm(inputs)
        x = inputs + self.ffn(x)
        return x

class BaseAttention(layers.Layer):
    """Base Attention Module.
    Args:
        num_heads (`int`): Number of attention heads.
        key_dim (`int`): Size of each attention head for key.
        dropout (`float`): Dropout probability for attention module.
    """

    def __init__(self, num_heads, key_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            name="mha",
        )
        self.query_layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="q_layernorm",
        )
        self.key_layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="k_layernorm",
        )
        self.value_layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="v_layernorm",
        )

        self.attention_scores = None

    def call(self, input_query, key, value):
        # Apply the attention module.
        query = self.query_layernorm(input_query)
        key = self.key_layernorm(key)
        value = self.value_layernorm(value)
        (attention_outputs, attention_scores) = self.multi_head_attention(
            query=query,
            key=key,
            value=value,
            return_attention_scores=True,
        )

        # Save the attention scores for later visualization.
        self.attention_scores = attention_scores

        # Add the input to the attention output.
        x = input_query + attention_outputs
        return x
class AttentionWithFFN(layers.Layer):
    """Attention with Feed Forward Network.
    Args:
        ffn_dims (`int`): Number of units in FFN.
        ffn_dropout (`float`): Dropout probability for FFN.
        num_heads (`int`): Number of attention heads.
        key_dim (`int`): Size of each attention head for key.
        attn_dropout (`float`): Dropout probability for attention module.
    """

    def __init__(
        self,
        ffn_dims,
        ffn_dropout,
        num_heads,
        key_dim,
        attn_dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Create the layers.
        self.attention = BaseAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=attn_dropout,
            name="base_attn",
        )
        self.ffn = FeedForwardNetwork(
            dims=ffn_dims,
            dropout=ffn_dropout,
            name="ffn",
        )

        self.attention_scores = None

    def call(self, query, key, value):
        # Apply the attention module.
        x = self.attention(query, key, value)

        # Save the attention scores for later visualization.
        self.attention_scores = self.attention.attention_scores

        # Apply the FFN.
        x = self.ffn(x)
        return x


class CustomRecurrentCell(layers.Layer):
    """Custom Recurrent Cell.
    Args:
        chunk_size (`int`): Number of tokens in a chunk.
        r (`int`): One Cross Attention per **r** Self Attention.
        num_layers (`int`): Number of layers.
        ffn_dims (`int`): Number of units in FFN.
        ffn_dropout (`float`): Dropout probability for FFN.
        num_heads (`int`): Number of attention heads.
        key_dim (`int`): Size of each attention head for key.
        attn_dropout (`float`): Dropout probability for attention module.
    """

    def __init__(
        self,
        chunk_size,
        r,
        num_layers,
        ffn_dims,
        ffn_dropout,
        num_heads,
        key_dim,
        attn_dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Save the arguments.
        self.chunk_size = chunk_size
        self.r = r
        self.num_layers = num_layers
        self.ffn_dims = ffn_dims
        self.ffn_droput = ffn_dropout
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn_dropout = attn_dropout

        # Create the state_size and output_size. This is important for
        # custom recurrence logic.
        self.state_size = tf.TensorShape([chunk_size, ffn_dims])
        self.output_size = tf.TensorShape([chunk_size, ffn_dims])

        self.get_attention_scores = False
        self.attention_scores = []

        # Perceptual Module
        perceptual_module = list()
        for layer_idx in range(num_layers):
            perceptual_module.append(
                AttentionWithFFN(
                    ffn_dims=ffn_dims,
                    ffn_dropout=ffn_dropout,
                    num_heads=num_heads,
                    key_dim=key_dim,
                    attn_dropout=attn_dropout,
                    name=f"pm_self_attn_{layer_idx}",
                )
            )
            if layer_idx % r == 0:
                perceptual_module.append(
                    AttentionWithFFN(
                        ffn_dims=ffn_dims,
                        ffn_dropout=ffn_dropout,
                        num_heads=num_heads,
                        key_dim=key_dim,
                        attn_dropout=attn_dropout,
                        name=f"pm_cross_attn_ffn_{layer_idx}",
                    )
                )
        self.perceptual_module = perceptual_module

        # Temporal Latent Bottleneck Module
        self.tlb_module = AttentionWithFFN(
            ffn_dims=ffn_dims,
            ffn_dropout=ffn_dropout,
            num_heads=num_heads,
            key_dim=key_dim,
            attn_dropout=attn_dropout,
            name=f"tlb_cross_attn_ffn",
        )

    def call(self, inputs, states):
        # inputs => (batch, chunk_size, dims)
        # states => [(batch, chunk_size, units)]
        slow_stream = states[0]
        fast_stream = inputs

        for layer_idx, layer in enumerate(self.perceptual_module):
            fast_stream = layer(query=fast_stream, key=fast_stream, value=fast_stream)

            if layer_idx % self.r == 0:
                fast_stream = layer(
                    query=fast_stream, key=slow_stream, value=slow_stream
                )

        slow_stream = self.tlb_module(
            query=slow_stream, key=fast_stream, value=fast_stream
        )

        # Save the attention scores for later visualization.
        if self.get_attention_scores:
            self.attention_scores.append(self.tlb_module.attention_scores)

        return fast_stream, [slow_stream]

class TemporalLatentBottleneckModel(keras.Model):
    """Model Trainer.
    Args:
        patch_layer ([`keras.layers.Layer`](/api/layers/base_layer#layer-class)): Patching layer.
        custom_cell ([`keras.layers.Layer`](/api/layers/base_layer#layer-class)): Custom Recurrent Cell.
    """

    def __init__(self, patch_layer, custom_cell, **kwargs):
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.rnn = layers.RNN(custom_cell, name="rnn")
        self.gap = layers.GlobalAveragePooling1D(name="gap")
        self.head = layers.Dense(10, activation="softmax", dtype="float32", name="head")

    def call(self, inputs):
        x = self.patch_layer(inputs)
        x = self.rnn(x)
        x = self.gap(x)
        outputs = self.head(x)
        return outputs

# Build the model.
patch_layer = PatchEmbedding(
    image_size=(config["image_size"], config["image_size"]),
    patch_size=(config["patch_size"], config["patch_size"]),
    embed_dim=config["embed_dim"],
    chunk_size=config["chunk_size"],
)
custom_rnn_cell = CustomRecurrentCell(
    chunk_size=config["chunk_size"],
    r=config["r"],
    num_layers=config["num_layers"],
    ffn_dims=config["embed_dim"],
    ffn_dropout=config["ffn_drop"],
    num_heads=config["num_heads"],
    key_dim=config["embed_dim"],
    attn_dropout=config["attn_drop"],
)
model = TemporalLatentBottleneckModel(
    patch_layer=patch_layer,
    custom_cell=custom_rnn_cell,
)

optimizer = AdamW(
    learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    epochs=config["epochs"],
    validation_data=val_ds,
)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()