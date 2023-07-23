import logging
from flask import Flask, jsonify, request
from kubernetes import client, config
import numpy as np
import tensorflow as tf
import pandas as pd
#from gru import CustomModel as Model
import json

logging.basicConfig(level=logging.INFO)

config.load_incluster_config()

# Création d'un objet client Kubernetes
k8s_client = client.CoreV1Api()
pod_ip_ = str
app = Flask(__name__)

batch_size = 1
timesteps = 100
# features = 1086
features = 1629
nb_classes = 250
#model = Model(batch_size, timesteps, features, nb_classes)

interpreter = tf.lite.Interpreter(model_path="/app/encoder_generator2_96_64_4.tflite")
interpreter.allocate_tensors()
model = interpreter.get_signature_runner()

interpreter2 = tf.lite.Interpreter(model_path="/app/encoder_generator2_432_320_4.tflite")
interpreter2.allocate_tensors()
model2 = interpreter.get_signature_runner()

selected_colomns = pd.read_json("/app/inference_args.json", orient="index")
selected_colomns = selected_colomns.transpose().values
selected_colomns = np.reshape(selected_colomns, (164,))

# Read Character to Ordinal Encoding Mapping
with open(f'/app/character_to_prediction_index.json') as json_file:
    CHAR2ORD = json.load(json_file)

# Ordinal to Character Mapping
ORD2CHAR = {j:i for i,j in CHAR2ORD.items()}

# Output Predictions to string
def outputs2phrase(outputs):
    if outputs.ndim == 2:
        outputs = np.argmax(outputs, axis=1)

    return ''.join([ORD2CHAR.get(s, '') for s in outputs])


def convert_df_to_np(df):
    nbframes = len(df["frame"].unique())
    frames = np.zeros((nbframes, 543, 3))
    for step, (name, timestep) in enumerate(df.groupby("frame")):
        frames[step, :, :] = timestep[["x", "y", "z"]].values
    sequence = np.reshape(np.stack(frames), (len(frames), 1629))
    return sequence


# @app.route('/run', methods=['POST'])
# def run_model():
#     logging.info(request.files)
#     if request.method == 'POST':
#         # Récupérer le fichier envoyé
#         uploaded_file = request.files['file']
#         logging.info(uploaded_file)
#         # Enregistrer le fichier dans le répertoire /app
#         train = pd.read_csv("train.csv")
#         dict_sign = {}
#         for i, sign in enumerate(train["sign"].unique()):
#             dict_sign[i] = sign
#         df = pd.read_csv(uploaded_file)
#         seq = convert_df_to_np(df)
#
#         seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1]))
#         seqs = []
#
#         if seq.shape[1] < timesteps:
#             for i in range(0, seq.shape[1], timesteps):
#                 if i + timesteps > seq.shape[1]:
#                     logging.error(f"zero: { np.zeros((timesteps - (seq.shape[1] - i), features)).shape}")
#                     logging.error(f"seq: {seq[0, i:].shape}")
#                     seqs.append(np.reshape(
#                         np.concatenate(
#                             [seq[0, i:], np.zeros((timesteps - (seq.shape[1] - i), features))]
#                         ),
#                         (1, timesteps, features)))
#                 else:
#                     seqs.append(np.reshape(seq[0, i:i + timesteps], (1, timesteps, features)))
#         else:
#             zero = np.zeros((1, timesteps - (seq.shape[1] % timesteps), features))
#             seq = np.concatenate((seq, zero), axis=1)
#             for i in range(0, seq.shape[1], timesteps):
#                 seqs.append(np.reshape(seq[0, i:i + timesteps], (1, timesteps, features)))
#         # compute the prediction for each sub seq
#         print("seqs: ", len(seqs))
#         print("seqs[0]: ", seqs[0].shape)
#         for ses_ in seqs:
#             p = model(ses_)
#             print("prob: ", np.max(p))
#             print("classe", dict_sign[np.argmax(p)])
#
#         return jsonify({"result": dict_sign[np.argmax(p)]})



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Récupérer le fichier envoyé
        uploaded_file = request.files['file']

        df = pd.read_csv(uploaded_file)
        data = df.astype(np.float32).values
        demo_output = model(inputs=data)['outputs']
        # Convert to string
        demo_output = outputs2phrase(demo_output)

        demo_output2 = model2(inputs=data)['outputs']
        # Convert to string
        demo_output2 = outputs2phrase(demo_output2)

        return jsonify({"result1": demo_output, "result2": demo_output2})

@app.route('/')
def health():
    return "ok"


if __name__ == '__main__':
    #model(tf.zeros((batch_size, timesteps, features)))
    #model.load_weights("model1_pretrained.h5")

    app.run(host='0.0.0.0', port=5000, debug=True)
