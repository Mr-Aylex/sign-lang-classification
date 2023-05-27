import logging
from flask import Flask, jsonify, request
import requests
from kubernetes import client, config

logging.basicConfig(level=logging.INFO)
config.load_incluster_config()

# Création d'un objet client Kubernetes
k8s_client = client.CoreV1Api()
pod_ip_ = str
app = Flask(__name__)

STATUS = "ok"


@app.route('/', methods=['GET'])
def get_status():
    return jsonify({"status": STATUS})


@app.route('/bonjours', methods=['GET'])
def get_bonjours():
    return jsonify({"status": "bonjours"})


@app.route('/ap1', methods=['GET'])
def get_ap1():
    pod_ip_ = get_pod_ip('web-service-api1')
    req = requests.get(f"http://{pod_ip_}:4000/status")
    return jsonify({"status": req.json()["status"]})


@app.route('/sayhello', methods=['GET'])
def get_sayhello():
    pod_ip_ = get_pod_ip('web-service-api1')
    req = requests.get(f"http://{pod_ip_}:4000/bonjours")
    if req.json()["status"] == "bonjours":
        return jsonify({"status": "bonjours"})


@app.route('/download', methods=['POST', 'GET'])
def upload_file():
    logging.info(request.files)
    if request.method == 'POST':
        # Récupérer le fichier envoyé
        uploaded_file = request.files['file']
        logging.info(uploaded_file)
        # Enregistrer le fichier dans le répertoire /app
        uploaded_file.save('/app/fichier.txt')
        return 'Fichier reçu et enregistré avec succès'
    else:
        logging.error('Méthode non autorisée')
        return 'Méthode non autorisée'


@app.route('/readfile', methods=['GET'])
def read_file():
    with open("/app/fichier.txt", 'rb') as file:
        file_content = file.read()
    return file_content
        # Effectuer la requête POST vers l'API de destination


def get_pod_ip(service_name, namespace='default'):
    try:
        # # Récupérer les informations du service
        service_info = k8s_client.read_namespaced_service(service_name, namespace)

        # Récupérer l'adresse IP du pod à partir du service
        pod_ip = service_info.spec.cluster_ip
        return pod_ip
    except Exception as e:
        return f"Erreur : {str(e)}"


if __name__ == '__main__':
    pod_ip_ = get_pod_ip('web-service-api1')
    app.run(host='0.0.0.0', debug=True, port=5000)

# les frain à l'usage de scrum
# 1- la difficulté de la mise en place de scrum
# 4- la communication avec le client
# 5- l'implication du client
