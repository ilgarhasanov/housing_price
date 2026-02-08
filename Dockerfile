FROM python:3.12-slim
#container icerisinde harda run olur
WORKDIR /app 

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# hara gonder burdan app-a
COPY pyproject.toml /app/
COPY src /app/src
COPY configs /app/configs
COPY artifacts /app/artifacts

# Install - update et pipi. cacheda saxlama, elave yaddasdi.
RUN pip install --no-cache-dir -U pip\
 && pip install --no-cache-dir -e .


ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# portu qeyd et
EXPOSE 8000 

# uvicorn- bizim kodu servere cevirir; servicede app ishe dussun
CMD ["uvicorn", "housing_model.service:app", "--host", "0.0.0.0", "--port", "8000"]

#  -t tag demekdi ve modeli deyir
# docker build -t housing-model:latest .
# pytest -q    - nece test pass olur onu deyir