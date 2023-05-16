import keras
import tensorflow as tf
import numpy as np
import time

MODEL_PATH = "custom_model.h5"

imported = tf.saved_model.load(MODEL_PATH)

print(imported.signatures)

tf_model = imported.signatures["serving_default"]

print(tf_model.structured_outputs)

