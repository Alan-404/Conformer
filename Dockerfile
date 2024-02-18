FROM python:3.10

WORKDIR /stt

COPY ./requirements.txt /stt/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /stt/requirements.txt

COPY ./api.py /stt/api.py
COPY ./processing/processor.py /stt/processing/processor.py
COPY ./checkpoints/model.pt /stt/checkpoints/model.pt
COPY ./model /stt/model

CMD ["python3", "api.py", "--model", "./checkpoints/model.pt", "--host", "0.0.0.0"]