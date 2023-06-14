FROM python:3.10
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
COPY . /app
RUN ls
EXPOSE 5000
CMD ["python", "/app/api.py"]
