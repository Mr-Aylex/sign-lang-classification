import logging
import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from kubernetes import client, config
from io import BytesIO
import requests

logging.getLogger().setLevel(logging.INFO)
config.load_incluster_config()

# Création d'un objet client Kubernetes
k8s_client = client.CoreV1Api()
pod_ip_ = str
app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def format_df(df):
    # Reshape the DataFrame using pivot_table
    df = pd.DataFrame({'landmark_index': pd.to_numeric(df['landmark_index'], errors='coerce'),
                       'frame': pd.to_numeric(df['frame'], errors='coerce'),
                       'type': df['type'].astype(str),
                       'x': df['x'],
                       'y': df['y'],
                       'z': df['z']})

    df_pivot = df.pivot_table(index='frame', columns=['type', 'landmark_index'])

    # Flatten the MultiIndex column names
    df_pivot.columns = ['{}_{}_{}'.format(pos, type_val, point) for pos, type_val, point in df_pivot.columns]

    # Create the final DataFrame
    final_df = pd.DataFrame(df_pivot.to_records())

    with open('removed_columns.txt', 'r') as file:
        removed_columns = [column.strip() for column in file.readlines()]
    final_df.drop(columns=removed_columns, inplace=True)

    return final_df


pd.set_option('display.max_rows', None)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def get_right_hand_landmark_indexes(hand, frame):
    index_list = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
                  'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
                  'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                  'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

    right_hand_landmarks = []
    for part in index_list:
        if hand.right_hand_landmarks:
            right_hand_landmarks.append(
                [frame, f"{frame}-right_hand-{index_list.index(part)}", "right_hand", index_list.index(part),
                 hand.right_hand_landmarks.landmark[mp_holistic.HandLandmark(index_list.index(part))].x,
                 hand.right_hand_landmarks.landmark[mp_holistic.HandLandmark(index_list.index(part))].y,
                 hand.right_hand_landmarks.landmark[mp_holistic.HandLandmark(index_list.index(part))].z])

        else:
            right_hand_landmarks.append(
                [frame, f"{frame}-right_hand-{index_list.index(part)}", "right_hand", index_list.index(part), 0, 0, 0])

    return right_hand_landmarks


def get_left_hand_landmark_indexes(hand, frame):
    index_list = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
                  'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
                  'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                  'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

    left_hand_landmarks = []
    for part in index_list:
        if hand.left_hand_landmarks:
            left_hand_landmarks.append(
                [frame, f"{frame}-left_hand-{index_list.index(part)}", "left_hand", index_list.index(part),
                 hand.left_hand_landmarks.landmark[mp_holistic.HandLandmark(index_list.index(part))].x,
                 hand.left_hand_landmarks.landmark[mp_holistic.HandLandmark(index_list.index(part))].y,
                 hand.left_hand_landmarks.landmark[mp_holistic.HandLandmark(index_list.index(part))].z])

        else:
            left_hand_landmarks.append(
                [frame, f"{frame}-left_hand-{index_list.index(part)}", "left_hand", index_list.index(part), 0, 0, 0])

    return left_hand_landmarks


def get_post_landmark_indexes(pose, frame):
    index_list = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
                  'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER',
                  'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY',
                  'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
                  'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX',
                  'RIGHT_FOOT_INDEX']

    pose_landmarks = []
    for part in index_list:
        if pose.pose_landmarks:
            pose_landmarks.append([frame, f"{frame}-pose-{index_list.index(part)}", "pose", index_list.index(part),
                                   pose.pose_landmarks.landmark[mp_holistic.PoseLandmark(index_list.index(part))].x,
                                   pose.pose_landmarks.landmark[mp_holistic.PoseLandmark(index_list.index(part))].y,
                                   pose.pose_landmarks.landmark[mp_holistic.PoseLandmark(index_list.index(part))].z])

        else:
            pose_landmarks.append(
                [frame, f"{frame}-pose-{index_list.index(part)}", "pose", index_list.index(part), 0, 0, 0])

    return pose_landmarks


def get_face_landmark_indexes(face, frame):
    face_landmarks = []
    if face.face_landmarks:
        for i in range(len(face.face_landmarks.landmark)):
            face_landmarks.append([frame, f"{frame}-face-{i}", "face", i, face.face_landmarks.landmark[i].x,
                                   face.face_landmarks.landmark[i].y, face.face_landmarks.landmark[i].z])
    else:
        for i in range(0, 468):
            face_landmarks.append([frame, f"{frame}-face-{i}", "face", i, 0, 0, 0])
    return face_landmarks


def Mediapipe_holistic(video):
    count = 0
    data = []
    cap = cv2.VideoCapture(video)
    # cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():

            success, image = cap.read()

            start = time.time()

            if image is not None:
                img_h, img_w, img_c = image.shape

                # Convert the BGR image to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False

                # Process the image and find hands
                results = hands.process(image)

                image.flags.writeable = True

                # Draw the hand annotations on the image.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                end = time.time()
                totalTime = end - start

                fps = 1 / totalTime
                count = count + 1

                data.append(get_right_hand_landmark_indexes(results, count))
                data.append(get_left_hand_landmark_indexes(results, count))
                data.append(get_post_landmark_indexes(results, count))
                data.append(get_face_landmark_indexes(results, count))
            else:
                cap.release()

            # for live video
            # cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            #
            # cv2.imshow('MediaPipe Hands', image)
            #
            # if cv2.waitKey(5) & 0xFF == 27:
            #   break
    flat_list = [item for sublist in data for item in sublist]
    df = pd.DataFrame(flat_list, columns=['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z'])

    df = format_df(df)


    return df


def get_pod_ip(service_name, namespace='default'):
    try:
        # Récupérer les informations du service
        service_info = k8s_client.read_namespaced_service(service_name, namespace)

        # Récupérer l'adresse IP du pod à partir du service
        pod_ip = service_info.spec.cluster_ip
        return pod_ip
    except Exception as e:
        return f"Erreur : {str(e)}"


@app.route('/')
def health():
    return 'OK'


@app.route('/run/<name>', methods=['GET'])
def run_mpm(name):
    logging.error(name)
    pod_ip_ = get_pod_ip('signaify-service')

    logging.error(name)
    df = Mediapipe_holistic(f"/mnt/data/{name}")
    df_buffer = BytesIO()
    df.to_csv(df_buffer, index=False)

    req_res = requests.post(f'http://{pod_ip_}:5000/run', files={'file': df_buffer.getvalue()})
    logging.info(req_res.text)
    return req_res.text


if __name__ == '__main__':
    pod_ip_ = get_pod_ip('signaify-service')
    app.run(host='0.0.0.0', port=4000, debug=True)
