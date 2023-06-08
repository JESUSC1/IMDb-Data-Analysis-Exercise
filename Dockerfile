# Dockerfile
FROM python:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY Data-Science-Exercise-Part-1.ipynb .

# Mount a volume to the container
VOLUME /app

CMD jupyter notebook Data-Science-Exercise-Part-1
