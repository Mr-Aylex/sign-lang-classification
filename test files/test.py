import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#for edge in FACEMESH_LIPS:



def read_video():
    count = 0
    data = []
    cap = cv2.VideoCapture("WIN_20230427_12_27_08_Pro.mp4")
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
            else:
                cap.release()

            for i in range(len(results.face_landmarks.landmark)):
                x = results.face_landmarks.landmark[i].x
                y = results.face_landmarks.landmark[i].y
                z = results.face_landmarks.landmark[i].z
                print(i)
                print(f"Landmark {i}: x={x}, y={y}, z={z}")


            # lips_landmarks = []
            # landmark_maps = dict()
            # for edge in FACEMESH_CONTOURS:
            #     landmark_maps[edge[0]] = results.face_landmarks.landmark[edge[0]]
            #     landmark_maps[edge[1]] = results.face_landmarks.landmark[edge[1]]
            #     landmark1 = results.face_landmarks.landmark[edge[0]]
            #     landmark2 = results.face_landmarks.landmark[edge[1]]
            #     lips_landmarks.append([landmark1.x, landmark1.y, landmark1.z])
            #     lips_landmarks.append([landmark2.x, landmark2.y, landmark2.z])



            count = count + 1
            print(count)

            #data.append(get_right_hand_landmark_indexes(results, count))

            # cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            #
            # cv2.imshow('MediaPipe Hands', image)
            #
            # if cv2.waitKey(5) & 0xFF == 27:
            #   break

    #



read_video()



# print(lips_landmarks)
# # plot x, y, and z coordinates of landmarks
# #x = [landmark[0] for landmark in lips_landmarks]
# #y = [landmark[1] for landmark in lips_landmarks]
# x= []
# y= []
#
# for k in lips_landmarks:
#     plt.annotate(k, (lips_landmarks[k].x, lips_landmarks[k].y))
#     x.append(lips_landmarks[k][0])
#     y.append(lips_landmarks[k][1])
#
# # set plot labels
# plt.scatter(x, y)
# plt.show()