import logging
from flask import Flask, jsonify, request
import requests

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)

STATUS = "ok"


@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"status": STATUS})


@app.route('/bonjours', methods=['GET'])
def get_bonjours():
    return jsonify({"status": "bonjours"})


@app.route('/ap2', methods=['GET'])
def get_ap2():
    req = requests.get(f"http://127.0.0.1:5000/status")
    return jsonify({"status": req.json()["status"]})


@app.route('/sayhello', methods=['GET'])
def get_sayhello():
    req = requests.get(f"http://127.0.0.1:5000/bonjours")
    if req.json()["status"] == "bonjours":
        return jsonify({"status": "bonjours"})

@app.route('/readfile', methods=['GET'])
def read_file():
    with open("/path/data/nom_du_fichier.txt", 'rb') as file:
        file_content = file.read()
    return file_content
        # Effectuer la requête POST vers l'API de destination


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # Configurer les informations de l'API de destination
    api_url = f"http://127.0.0.1:5000/download"
    headers = {'Content-Type': 'multipart/form-data'}

    with open("/app/fichier.txt", 'rb') as file:
        logging.info(file)
        # Effectuer la requête POST vers l'API de destination
        response = requests.post(api_url, headers=headers)
    logging.info(response.text)
    if response.status_code == 200:
        return 'Fichier envoyé avec succès'
    else:
        return 'Une erreur s\'est produite lors de l\'envoi du fichier'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)
