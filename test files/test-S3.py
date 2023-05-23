#import mysql.connector
import pandas as pd
from tqdm import tqdm
import boto3 as bt
import os
import requests
import logging
import cv2


logging.basicConfig(level=logging.INFO)

from botocore.config import Config

CREDENTIAL_FILE = pd.read_csv("../S3/Theo-Dalex_credentials.csv")
ACCESS_KEY = CREDENTIAL_FILE['Nom d\'utilisateur'][0]
SECRET_KEY = CREDENTIAL_FILE['Mot de passe'][0]


BUCKET_NAME = "sign-video"

S3 = bt.resource(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(
        region_name='eu-west-3',
        )
)


def download_file(bucket_name, file_name):
    logging.info('Downloading file %s from bucket %s', file_name, bucket_name)
    S3.Bucket(bucket_name).download_file(Key=file_name, Filename=file_name)
    logging.info('Download complete')

def delete_all_files(bucket_name):
    logging.info('Deleting all files from bucket %s', bucket_name)
    bucket = S3.Bucket(bucket_name)
    bucket.objects.all().delete()
    logging.info('Delete complete')


for object in S3.Bucket(BUCKET_NAME).objects.all():
    print(object)


