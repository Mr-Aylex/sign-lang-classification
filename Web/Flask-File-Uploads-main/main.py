import logging
import boto3 as bt
import pandas as pd
import uuid
from flask import Flask, render_template, request, url_for, redirect, jsonify
from flask_wtf import FlaskForm
from kubernetes.client import ApiException
from wtforms import FileField, SubmitField
from botocore.config import Config
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from kubernetes import client, config
import json
import time
import requests
import binascii
import gzip
import base64

app = Flask(__name__)
CREDENTIAL_FILE = pd.read_csv("/app/Theo-Dalex_credentials.csv")
ACCESS_KEY = CREDENTIAL_FILE['Nom d\'utilisateur'][0]
SECRET_KEY = CREDENTIAL_FILE['Mot de passe'][0]
STORAGE_PATH = "/path/data/"
BUCKET_NAME = "sign-video"
df = pd.DataFrame()

@app.route('/dataframe', methods=['GET'])
def get_dataframe():
    global df
    # Generate or fetch the big dataframe
    big_dataframe = df

    # Convert the dataframe to JSON
    dataframe_json = big_dataframe.to_json()

    # Return the JSON response
    return jsonify(dataframe_json)

def wait_for_pod_completion(api_client, pod_name, namespace):
    while True:
        try:
            pod = api_client.read_namespaced_pod(pod_name, namespace)
            if pod.status.phase in ["Running", "Succeeded"]:
                break  # Pod is in the desired state, exit the loop
        except ApiException as e:
            print("Exception when retrieving pod status: %s\n" % e)
        time.sleep(25)


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET', "POST"])
def index():
    return render_template('index.html')


ALLOWED_EXTENSIONS = {'mp4'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', "POST"])
def upload():
    if 'video' not in request.files:
        return "No video file found"
    video = request.files['video']
    if video.filename == '':
        return "No video file selected"
    if video and allowed_file(video.filename):
        """video.save('static/files/'+video.filename)"""
        new_filename = uuid.uuid4().hex + '.' + video.filename.rsplit('.', 1)[1].lower()
        bucket_name = "sign-video"
        s3 = bt.resource(
            's3',
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(
                region_name='eu-west-3',
            )
        )
        video_url = f'https://{bucket_name}.s3.amazonaws.com/{video.filename}'
        logging.error(video_url)
        logging.error(video.filename)
        logging.error(bucket_name)
        video.seek(0)
        s3.Bucket(bucket_name).upload_fileobj(video, video.filename)

        return render_template('preview.html', video_name=video.filename, video_url=video_url)

    return "pas le bon type de fichier"


@app.route('/translate', methods=['GET', "POST"])
def translate():
    config.load_incluster_config()

    # Create a Kubernetes API client
    api_client = client.CoreV1Api()

    # Define the Pod specification
    pod_manifest = {
        'apiVersion': 'v1',
        'kind': 'Pod',
        'metadata': {
            'name': 'mpm'
        },
        'spec': {
            'restartPolicy': 'Never',
            'containers': [
                {
                    'name': 'mpm',
                    'image': 'dopehat54/mpm:latest',
                    'ports': [{'containerPort': 80}]
                }
            ]
        }
    }
    try:
        api_client.read_namespaced_pod(name="mpm", namespace="default")
        # Pod exists, so delete it
        api_client.delete_namespaced_pod(name="mpm", namespace="default")
        print(f"Pod mpm in namespace default deleted successfully.")
    except client.rest.ApiException as e:
        if e.status == 404:
            # Pod doesn't exist, so do nothing
            print(f"Pod mpm in namespace default doesn't exist.")
        else:
            # Unexpected error occurred
            print(f"Error: {e}")
    api_client.create_namespaced_pod(namespace='default', body=pod_manifest) #création de pod
    wait_for_pod_completion(api_client, 'mpm', 'default')

    pod_data = api_client.read_namespaced_pod_log(name='mpm', namespace='default')
    filtered_lines = [line for line in pod_data.splitlines() if
                      "ERROR:root:test" not in line and "INFO: Created TensorFlow Lite XNNPACK delegate for CPU" not in line]
    lines = [line.split(',') for line in filtered_lines if line]

    if lines:
        columns = lines[0]
        data = lines[1:]
        global df
        df = pd.DataFrame(data, columns=columns)
    else:
        df = pd.DataFrame(columns=['Log'])

    df_json = df.to_json(orient='records')

    # Save JSON to file
    output_file = STORAGE_PATH + 'data.json'
    with open(output_file, 'w') as file:
        file.write(df_json)

    print(f"DataFrame saved as JSON at {output_file}")
    # print(df)
    # compressed_data = gzip.compress(df.to_json(orient='records').encode('utf-8'))
    #
    # encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    prediction_pod_manifest = {
        'apiVersion': 'v1',
        'kind': 'Pod',
        'metadata': {
            'name': 'signaify'
        },
        'spec': {
            'restartPolicy': 'Never',
            'containers': [
                {
                    'name': 'signaify',
                    'image': 'dopehat54/signaify:latest',
                    'ports': [{'containerPort': 90}],
                    'resources': {
                        'limits': {
                            'cpu': '2',
                            'memory': '2Gi'
                        }
                    },
                    'volumeMounts': [
                        {
                            'mountPath': '/path/data/',
                            'name': 'my-volume'
                        }
                    ]
                }
            ],
            'volumes': [
                {
                    'name': 'my-volume',
                    "persistentVolumeClaim": {
                        "claimName": "my-pvc"
                    }
                }
            ]
        }
    }
    try:
        api_client.read_namespaced_pod(name='signaify', namespace='default')
        # Pod exists, so delete it
        api_client.delete_namespaced_pod(name='signaify', namespace='default')
        print(f'Pod signaify in namespace default deleted successfully.')
    except client.rest.ApiException as e:
        if e.status == 404:
            # Pod doesn't exist, so do nothing
            print(f'Pod signaify in namespace default doesn\'t exist.')
        else:
            # Unexpected error occurred
            print(f'Error: {e}')

    api_client.create_namespaced_pod(namespace='default', body=prediction_pod_manifest)  # Create prediction Pod

    # Wait for the prediction Pod to be ready
    pod_ready = False
    while not pod_ready:
        try:
            pod_info = api_client.read_namespaced_pod(name='signaify', namespace='default')
            if pod_info.status.phase == 'Running':
                pod_ready = True
        except client.rest.ApiException as e:
            if e.status == 404:
                # Pod not found, keep waiting
                continue
            else:
                # Unexpected error occurred
                print(f'Error: {e}')
                break

    if pod_ready:
        # Get the prediction result from the Pod
        prediction_result = requests.get('http://localhost:90')
        prediction = json.loads(prediction_result.text)
        # Do something with the prediction...

    return 'Prediction completed.'


if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.run(host='0.0.0.0',debug=True, port=6001)
