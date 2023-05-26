import logging
from flask import Flask, jsonify, request
import requests
from kubernetes import client, config

logging.basicConfig(level=logging.INFO)

#config.load_incluster_config()

# Création d'un objet client Kubernetes
#k8s_client = client.CoreV1Api()
pod_ip_ = str
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
    pod_ip_ = get_pod_ip('web-service-api2')
    req = requests.get(f"http://{pod_ip_}:5000/status")
    return jsonify({"status": req.json()["status"]})


@app.route('/sayhello', methods=['GET'])
def get_sayhello():
    pod_ip_ = get_pod_ip('web-service-api2')
    req = requests.get(f"http://{pod_ip_}:5000/bonjours")
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
    pod_ip_ = get_pod_ip('web-service-api2')
    # Configurer les informations de l'API de destination
    api_url = f"http://{pod_ip_}:5000/download"
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


def get_pod_ip(service_name, namespace='default'):


    try:
        # Récupérer les informations du service
        # service_info = k8s_client.read_namespaced_service(service_name, namespace)
        #
        # # Récupérer l'adresse IP du pod à partir du service
        # pod_ip = service_info.spec.cluster_ip
        pod_ip = "127.0.0.1"
        return pod_ip
    except Exception as e:
        return f"Erreur : {str(e)}"


if __name__ == '__main__':
    pod_ip_ = get_pod_ip('web-service-api2')
    app.run(host='0.0.0.0', debug=True, port=4000)
