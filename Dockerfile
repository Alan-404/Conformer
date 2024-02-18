FROM python:3.10

WORKDIR /stt

COPY ./requirements.txt /stt/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /stt/requirements.txt

COPY ./api.py /stt/api.py
COPY ./vocabulary/dictionary.json /stt/vocabulary/dictionary.json
COPY ./processing/processor.py /stt/processing/processor.py
COPY ./lm/4gram.arpa /stt/lm/4gram.arpa
COPY ./checkpoints/model.pt /stt/checkpoints/model.pt
COPY ./model /stt/model

CMD ["python3", "api.py", "--model", "./checkpoints/model.pt", "--host", "0.0.0.0"]