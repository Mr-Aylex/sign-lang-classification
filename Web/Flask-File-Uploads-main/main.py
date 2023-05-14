from flask import Flask, render_template,request 
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from botocore.config import Config
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from kubernetes import client, config

app = Flask(__name__, template_folder='templates')

app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
def index():
    return render_template("index.html")
ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET',"POST"])
def upload():
    if 'video' not in request.files:
       return "No video file found"
    video = request.files['video']
    if video.filename == '':
        return "No video file selected"
    if video and allowed_file(video.filename):
        #video.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video.filename)))
        video.save('static/files/'+video.filename)
        return render_template('preview.html', video_name = video.filename)
    return "pas le bon type de fichier"


@app.route('/translate', methods=['GET', "POST"])
def translate():
    config.load_kube_config()

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
    return "Pod created"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)