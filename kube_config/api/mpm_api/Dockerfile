FROM python:3.10
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 4000
CMD ["python", "/app/api.py"]