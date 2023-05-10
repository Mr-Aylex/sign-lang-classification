from flask import Flask, render_template,request 
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)