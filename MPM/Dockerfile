FROM python:3.10
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
CMD ["python", "/app/Mediapipe_holistic_model.py"]