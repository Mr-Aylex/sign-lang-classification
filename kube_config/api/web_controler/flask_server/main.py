import logging
import uuid
import boto3 as bt
from botocore.config import Config
import pandas as pd
import requests
from flask import Flask, render_template, request
from kubernetes import client, config

logging.basicConfig(level=logging.INFO)

config.load_incluster_config()

# Création d'un objet client Kubernetes
k8s_client = client.CoreV1Api()

app = Flask(__name__)
CREDENTIAL_FILE = pd.read_csv("../signaify_accessKeys.csv")
ACCESS_KEY = CREDENTIAL_FILE['Access key ID'][0]
SECRET_KEY = CREDENTIAL_FILE['Secret access key'][0]
BUCKET_NAME = "signaify"
ALLOWED_EXTENSIONS = {'mp4'}

@app.route('/', methods=['GET', "POST"])
def index():
    return render_template('index.html')

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
    if video and video.filename.endswith(".mp4"):

        new_filename = f"{uuid.uuid4()}.mp4"
        video.save(f"/mnt/data/{new_filename}")

        bucket_name = BUCKET_NAME
        s3 = bt.resource(
            's3',
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(
                region_name='eu-west-1',
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


def get_pod_ip(service_name, namespace='default'):
    try:
        # Récupérer les informations du service
        service_info = k8s_client.read_namespaced_service(service_name, namespace)

        # Récupérer l'adresse IP du pod à partir du service
        pod_ip = service_info.spec.cluster_ip
        return pod_ip
    except Exception as e:
        return f"Erreur : {str(e)}"


def refresh_pod_ip():
    global pod_ip_
    pod_ip_ = dict()
    pod_ip_['mpm_api'] = get_pod_ip('mpm-api-service')
    pod_ip_['signaify'] = get_pod_ip('signaify-service')


@app.route('/translate', methods=['GET', "POST"])
def translate():
    refresh_pod_ip()
    config.load_incluster_config()

    # Create a Kubernetes API client
    api_client = client.CoreV1Api()

    # get id of the video
    video_id = request.form.get('video_name')
    # load the video from local storage to memory


    

    # Send the video name to the API
    req_res = requests.get(f"http://{pod_ip_['mpm_api']}:4000/run/{video_id}")

    
    logging.info(req_res.text)
    # Get the translated video from the API
    #translated_video = requests.get(f"http://{pod_ip_['signaify']}:5000/video/{video_id}")

    # Save the translated video to local storage
    # with open(f"/mnt/data/{video_id}", 'wb') as f:
    #     f.write(translated_video.content)

    return req_res.text


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=6000)
