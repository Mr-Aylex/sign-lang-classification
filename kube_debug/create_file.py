import numpy as np
from kubernetes import client, config
import requests
import json
import logging

#seq = np.zeros((1, 800, 523, 3))

#np.save("/path/data/seq.npy", seq)

config.load_incluster_config()
seq = np.load("/path/data/seq.npy")

logging.error(seq.shape)
# Create a Kubernetes API client
# api_client = client.CoreV1Api()
#
# pod_manifest = {
#     "kind": "Pod",
#     "metadata": {
#       "name": "namepod"
#     },
#     "spec": {
#         "containers": [
#             {
#                 "name": "test",
#                 "image": "nginx:latest"
#             }
#         ]
#     }
# }
#
# # Create a pod configuration object
# pod_config = client.V1Pod(**pod_manifest)
#
# api_instance = client.CoreV1Api()
# api_instance.create_namespaced_pod(body=pod_config, namespace="default")


