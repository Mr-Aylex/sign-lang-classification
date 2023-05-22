FROM python:3.10
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "/app/Run_model.py"]