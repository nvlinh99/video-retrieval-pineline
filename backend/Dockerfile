FROM continuumio/miniconda3

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

COPY utils utils

COPY main.py main.py

EXPOSE 8080

CMD ["python", "main.py"]