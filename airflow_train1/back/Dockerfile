FROM apache/spark:3.4.2-scala2.12-java11-python3-ubuntu
USER root

COPY model_predict.py .
COPY my_xgb_model.joblib .
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

ENTRYPOINT ["uvicorn","model_predict:app","--host","0.0.0.0","--port","8000"]