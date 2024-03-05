FROM python:3.10-slim
WORKDIR /app

#fixinf problem with libgomp.so.1: cannot open shared object file:
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

COPY main.py .
COPY model_new.h5 .
COPY resnet50_model.h5 .
COPY requirements.txt .


RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]