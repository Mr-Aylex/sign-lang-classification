FROM python:3.10
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY Flask-File-Uploads-main /app

EXPOSE 5000
CMD ["python", "/app/main.py"]
